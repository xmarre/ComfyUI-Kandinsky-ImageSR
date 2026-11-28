#!/usr/bin/env python3
"""Benchmark the native pipeline inside a ComfyUI environment (no server).

Measures model load time (from disk into this process; the operating system's file
cache may already hold the weights, so drop it beforehand for a true cold start),
warm end-to-end latency, tile throughput, per-stage time (VAE encode, transformer,
VAE decode; measured with device synchronisation, so
stage timing is only collected when ``--stages`` is given) and peak device memory.

Memory is reported from two sources, kept separate: PyTorch's caching allocator
(``peak_allocated_gib``/``peak_reserved_gib``) and device-wide usage sampled from
the driver (``device_used_*_gib``; total minus free, so it includes other
processes, CUDA context and allocations outside PyTorch's allocator such as those
of ComfyUI's dynamic VRAM manager). ``transformer_loaded_gib`` is the transformer
weight memory ComfyUI reports as resident on the device after each run. Both
models are loaded once before the first configuration and stay loaded unless
``--unload-between-configs`` is given; without it, a configuration that does
not need the transformer (correction strength 0) still runs with it resident.
``transformer_evaluations`` counts tiles whose transformer result reached the
output; ``transformer_tiles_submitted`` also counts work discarded by
out-of-memory retries.

Example::

    python scripts/benchmark.py --comfyui /path/to/ComfyUI \
        --model /models/kandinsky_imagesr_2.0.safetensors --vae /models/vae/ae.safetensors \
        --sizes 1024 2048 --factors 2 4 --tile-batch-sizes 1 2 4 0 --accumulation auto cpu \
        --comfy-args --use-pytorch-cross-attention

Attention backends are selected with ComfyUI's own flags (``--use-pytorch-cross-attention``,
``--use-sage-attention``, ``--use-flash-attention``, ...); run the script once per backend.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DeviceMemorySampler:
    """Peak device-wide memory in use (driver view), sampled in a background thread."""

    def __init__(self, device: torch.device, interval: float = 0.02) -> None:
        self.device, self.interval = device, interval
        self.peak = self.start_used = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def used(self) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        return total - free

    def __enter__(self) -> DeviceMemorySampler:
        self.start_used = self.peak = self.used()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.used())

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.used())


def _loaded_bytes(patcher) -> int | None:
    loaded = getattr(patcher, "loaded_size", None)
    return int(loaded()) if callable(loaded) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", required=True)
    parser.add_argument(
        "--model", "--bundle", dest="model", required=True, help="single-file checkpoint or official bundle directory"
    )
    parser.add_argument("--vae", help="FLUX.1 ae.safetensors (required for a single-file checkpoint)")
    parser.add_argument("--vae-dtype", default="default", choices=["default", "bf16", "fp32"])
    parser.add_argument("--patcher", default="standard", choices=["standard", "dynamic"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024], help="square input sizes")
    parser.add_argument("--factors", type=int, nargs="+", default=[2], choices=[2, 4])
    parser.add_argument("--creativity", type=int, default=4)
    parser.add_argument("--tile-batch-sizes", type=int, nargs="+", default=[1, 0], help="0 = automatic")
    parser.add_argument("--accumulation", nargs="+", default=["auto"], choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--output-scales",
        type=float,
        nargs="+",
        default=[0.0],
        help="final output scales (0 = native size); a fractional scale adds the final resize to the timing",
    )
    parser.add_argument("--weight-dtype", default="bf16", choices=["default", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="measured repetitions (at least 1)")
    parser.add_argument(
        "--restoration-modes", nargs="+", default=["calibrated"], choices=["calibrated", "direct_fractional"]
    )
    parser.add_argument("--correction-strengths", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--unload-between-configs",
        action="store_true",
        help="unload all models before each configuration, so memory residency reflects only what that "
        "configuration loads (for example no transformer at correction strength 0); the first repetition "
        "then includes loading, so use --warmup 1 or more for timing",
    )
    parser.add_argument("--reconstructions", nargs="+", default=["latent"], choices=["latent", "residual"])
    parser.add_argument(
        "--stages",
        action="store_true",
        help="time input sampling, encode, transformer, decode, accumulation and output assembly "
        "(adds device synchronisation; use separate runs without it for throughput)",
    )
    parser.add_argument("--json", help="write results as JSON")
    parser.add_argument("--comfy-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be at least 1 and --warmup non-negative")

    from _comfy_bootstrap import bootstrap

    bootstrap(args.comfyui, args.comfy_args)
    import comfy.model_management as mm
    from _comfy_bootstrap import load_flux_vae

    from kandinsky_sr import runtime

    device = mm.get_torch_device()
    start = time.perf_counter()
    model = runtime.load_model(
        args.model,
        Path(args.model).name,
        weight_dtype=args.weight_dtype,
        vae_dtype=args.vae_dtype,
        patcher=args.patcher,
    )
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vae = load_flux_vae(args.vae, args.vae_dtype) or model.bundled_vae()
    vae_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mm.load_models_gpu([model.patcher, vae.patcher])
    _sync(device)
    to_device_seconds = time.perf_counter() - start
    print(
        f"cold load: transformer {load_seconds:.2f}s and VAE {vae_load_seconds:.2f}s from disk, "
        f"{to_device_seconds:.2f}s to {device} ({type(model.patcher).__name__})"
    )

    from kandinsky_sr.direct import DirectRestorer
    from kandinsky_sr.pipeline import TiledRestorer

    stage_times: dict[str, list[float]] = {"encode": [], "transform": [], "decode": []}

    def instrument(owner, attribute: str, stage: str) -> None:
        """Time ``owner.attribute`` (synchronised) into ``stage_times[stage]``."""
        original = getattr(owner, attribute)
        stage_times.setdefault(stage, [])

        def timed(*a, **kw):
            _sync(device)
            t = time.perf_counter()
            out = original(*a, **kw)
            _sync(device)
            stage_times[stage].append(time.perf_counter() - t)
            return out

        setattr(owner, attribute, timed)

    if args.stages:
        for name in ("encode", "transform", "decode"):
            instrument(runtime._ComfyBackend, name, name)
        # input tiles: native bicubic enlargement or direct sampling
        instrument(TiledRestorer, "_model_pixels", "input_sampling")
        instrument(DirectRestorer, "_model_pixels", "input_sampling")
        instrument(TiledRestorer, "_commit", "accumulation")
        instrument(TiledRestorer, "_finalize", "accumulation")
        # output assembly includes the final resize when there is one
        instrument(runtime, "assemble_output", "output_assembly")
        if any(scale for scale in args.output_scales):
            instrument(runtime, "_finish_resized", "final_resize")

    results = {
        "device": str(device),
        "torch": torch.__version__,
        "patcher": type(model.patcher).__name__,
        "load_seconds": load_seconds,
        "vae_load_seconds": vae_load_seconds,
        "to_device_seconds": to_device_seconds,
        "runs": [],
    }
    gen = torch.Generator().manual_seed(0)
    for size in args.sizes:
        image = torch.rand(1, size, size, 3, generator=gen)
        for factor in args.factors:
            for batch, acc, output_scale, mode, strength, reconstruction in (
                (b, a, o, m, cs, r)
                for b in args.tile_batch_sizes
                for a in args.accumulation
                for o in args.output_scales
                for m in args.restoration_modes
                for cs in args.correction_strengths
                for r in args.reconstructions
                # calibrated output scales above the native factor are invalid for it
                if o <= (4 if m == "direct_fractional" else factor)
            ):
                repetitions = []
                if args.unload_between_configs:
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                for k in range(args.warmup + args.repeats):
                    for v in stage_times.values():
                        v.clear()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    _sync(device)
                    sampler = DeviceMemorySampler(device) if device.type == "cuda" else None
                    if sampler is not None:
                        sampler.__enter__()
                    t = time.perf_counter()
                    try:
                        output, stats = runtime.upscale(
                            model,
                            image,
                            factor,
                            args.creativity,
                            tile_batch_size=batch,
                            accumulation_device=acc,
                            vae=vae,
                            output_scale=output_scale,
                            restoration_mode=mode,
                            correction_strength=strength,
                            reconstruction=reconstruction,
                        )
                        _sync(device)
                        seconds = time.perf_counter() - t
                    finally:
                        if sampler is not None:
                            sampler.__exit__(None, None, None)
                    output_device = str(output.device)
                    # Release the output before the next repetition so it cannot
                    # reduce the memory available to (and the batch chosen by) it.
                    del output
                    if k < args.warmup:
                        continue
                    rep = {
                        "seconds": seconds,
                        "output_size": list(stats.output_size),
                        "output_device": output_device,
                        # final assembly ran out of device memory and finished in system memory
                        "assembly_fallback": stats.assembly_fallback,
                        "tiles": stats.tiles,
                        "transformer_evaluations": stats.transformer_evaluations,
                        "transformer_tiles_submitted": stats.transformer_tiles_submitted,
                        "effective_batch": stats.final_batch_size,
                        "batch_sizes": sorted(set(stats.batch_sizes)),
                        "oom_retries": stats.oom_retries,
                    }
                    if device.type == "cuda":
                        rep["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                        rep["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
                        rep["device_used_before_gib"] = sampler.start_used / 1024**3
                        rep["device_used_peak_gib"] = sampler.peak / 1024**3
                    loaded = _loaded_bytes(model.patcher)
                    if loaded is not None:
                        rep["transformer_loaded_gib"] = loaded / 1024**3
                    if args.stages:
                        rep["stage_seconds"] = {name: sum(v) for name, v in stage_times.items()}
                    repetitions.append(rep)
                times = [r["seconds"] for r in repetitions]
                median = statistics.median(times)
                entry = {
                    "input": size,
                    "factor": factor,
                    "tiles": repetitions[0]["tiles"],
                    "tile_batch_size": batch,
                    "accumulation": acc,
                    "output_scale": output_scale,
                    "restoration_mode": mode,
                    "correction_strength": strength,
                    "reconstruction": reconstruction,
                    "median_seconds": median,
                    "min_seconds": min(times),
                    "tiles_per_second": repetitions[0]["tiles"] / median,
                    # aggregated over all measured repetitions, not taken from the last one
                    "effective_batches": sorted({r["effective_batch"] for r in repetitions}),
                    "oom_retries_total": sum(r["oom_retries"] for r in repetitions),
                    "assembly_fallbacks": sum(r["assembly_fallback"] for r in repetitions),
                    "repetitions": repetitions,
                }
                if device.type == "cuda":
                    entry["max_peak_allocated_gib"] = max(r["peak_allocated_gib"] for r in repetitions)
                    entry["max_peak_reserved_gib"] = max(r["peak_reserved_gib"] for r in repetitions)
                    entry["max_device_used_peak_gib"] = max(r["device_used_peak_gib"] for r in repetitions)
                if args.stages:
                    entry["median_stage_seconds"] = {
                        name: statistics.median(r["stage_seconds"][name] for r in repetitions) for name in stage_times
                    }
                results["runs"].append(entry)
                print(json.dumps(entry), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

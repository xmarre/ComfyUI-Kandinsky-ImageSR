#!/usr/bin/env python3
"""Benchmark the native pipeline inside a ComfyUI environment (no server).

Measures model load time (from disk into this process; the operating system's file
cache may already hold the weights, so drop it beforehand for a true cold start),
warm end-to-end latency, tile throughput, per-stage time (VAE encode, transformer,
VAE decode; measured with device synchronisation, so
stage timing is only collected when ``--stages`` is given) and peak device memory.

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
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    parser.add_argument("--weight-dtype", default="bf16", choices=["default", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="measured repetitions (at least 1)")
    parser.add_argument("--stages", action="store_true", help="time encode/transformer/decode (adds syncs)")
    parser.add_argument("--json", help="write results as JSON")
    parser.add_argument("--comfy-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be at least 1 and --warmup non-negative")

    from _comfy_bootstrap import bootstrap

    bootstrap(args.comfyui, args.comfy_args)
    import comfy.model_management as mm

    from kandinsky_sr import runtime

    device = mm.get_torch_device()
    start = time.perf_counter()
    from _comfy_bootstrap import load_flux_vae

    model = runtime.load_model(
        args.model,
        Path(args.model).name,
        weight_dtype=args.weight_dtype,
        vae_dtype=args.vae_dtype,
        patcher=args.patcher,
    )
    load_seconds = time.perf_counter() - start
    vae = load_flux_vae(args.vae, args.vae_dtype) or model.bundled_vae()
    mm.load_models_gpu([model.patcher, vae.patcher])
    _sync(device)
    to_device_seconds = time.perf_counter() - start - load_seconds
    print(f"cold load: {load_seconds:.2f}s from disk, {to_device_seconds:.2f}s to {device}")

    stage_times: dict[str, list[float]] = {"encode": [], "transform": [], "decode": []}
    if args.stages:
        cls = runtime._ComfyBackend
        for name in stage_times:
            original = getattr(cls, name)

            def timed(self, *a, _orig=original, _name=name):
                _sync(device)
                t = time.perf_counter()
                out = _orig(self, *a)
                _sync(device)
                stage_times[_name].append(time.perf_counter() - t)
                return out

            setattr(cls, name, timed)

    results = {
        "device": str(device),
        "torch": torch.__version__,
        "load_seconds": load_seconds,
        "to_device_seconds": to_device_seconds,
        "runs": [],
    }
    gen = torch.Generator().manual_seed(0)
    for size in args.sizes:
        image = torch.rand(1, size, size, 3, generator=gen)
        for factor in args.factors:
            for batch in args.tile_batch_sizes:
                for acc in args.accumulation:
                    repetitions = []
                    for k in range(args.warmup + args.repeats):
                        for v in stage_times.values():
                            v.clear()
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        _sync(device)
                        t = time.perf_counter()
                        output, stats = runtime.upscale(
                            model,
                            image,
                            factor,
                            args.creativity,
                            tile_batch_size=batch,
                            accumulation_device=acc,
                            vae=vae,
                        )
                        _sync(device)
                        seconds = time.perf_counter() - t
                        # Release the output before the next repetition so it cannot
                        # reduce the memory available to (and the batch chosen by) it.
                        del output
                        if k < args.warmup:
                            continue
                        rep = {
                            "seconds": seconds,
                            "tiles": stats.tiles,
                            "effective_batch": stats.final_batch_size,
                            "batch_sizes": sorted(set(stats.batch_sizes)),
                            "oom_retries": stats.oom_retries,
                        }
                        if device.type == "cuda":
                            rep["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                            rep["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
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
                        "median_seconds": median,
                        "min_seconds": min(times),
                        "tiles_per_second": repetitions[0]["tiles"] / median,
                        # aggregated over all measured repetitions, not taken from the last one
                        "effective_batches": sorted({r["effective_batch"] for r in repetitions}),
                        "oom_retries_total": sum(r["oom_retries"] for r in repetitions),
                        "repetitions": repetitions,
                    }
                    if device.type == "cuda":
                        entry["max_peak_allocated_gib"] = max(r["peak_allocated_gib"] for r in repetitions)
                        entry["max_peak_reserved_gib"] = max(r["peak_reserved_gib"] for r in repetitions)
                    if args.stages:
                        entry["median_stage_seconds"] = {
                            name: statistics.median(r["stage_seconds"][name] for r in repetitions)
                            for name in stage_times
                        }
                    results["runs"].append(entry)
                    print(json.dumps(entry), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

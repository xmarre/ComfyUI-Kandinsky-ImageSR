#!/usr/bin/env python3
"""Benchmark the native pipeline inside a ComfyUI environment (no server).

Measures model load time (from disk into this process; the operating system's file
cache may already hold the weights, so drop it beforehand for a true cold start),
warm end-to-end latency, tile throughput, per-stage time (VAE encode, transformer,
VAE decode; measured with device synchronisation, so
stage timing is only collected when ``--stages`` is given) and peak device memory.

Example::

    python scripts/benchmark.py --comfyui /path/to/ComfyUI --bundle /models/Kandinsky-ImageSR-2.0 \
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
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024], help="square input sizes")
    parser.add_argument("--factors", type=int, nargs="+", default=[2], choices=[2, 4])
    parser.add_argument("--creativity", type=int, default=4)
    parser.add_argument("--tile-batch-sizes", type=int, nargs="+", default=[1, 0], help="0 = automatic")
    parser.add_argument("--accumulation", nargs="+", default=["auto"], choices=["auto", "cpu", "gpu"])
    parser.add_argument("--weight-dtype", default="bf16", choices=["default", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stages", action="store_true", help="time encode/transformer/decode (adds syncs)")
    parser.add_argument("--json", help="write results as JSON")
    parser.add_argument("--comfy-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    from _comfy_bootstrap import bootstrap

    bootstrap(args.comfyui, args.comfy_args)
    import comfy.model_management as mm

    from kandinsky_sr import runtime

    device = mm.get_torch_device()
    start = time.perf_counter()
    model = runtime.load_model(args.bundle, Path(args.bundle).name, weight_dtype=args.weight_dtype)
    load_seconds = time.perf_counter() - start
    mm.load_models_gpu([model.patcher, model.vae.patcher])
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
                    times = []
                    for k in range(args.warmup + args.repeats):
                        for v in stage_times.values():
                            v.clear()
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        _sync(device)
                        t = time.perf_counter()
                        _, stats = runtime.upscale(
                            model, image, factor, args.creativity, tile_batch_size=batch, accumulation_device=acc
                        )
                        _sync(device)
                        if k >= args.warmup:
                            times.append(time.perf_counter() - t)
                    entry = {
                        "input": size,
                        "factor": factor,
                        "tiles": stats.tiles,
                        "tile_batch_size": batch,
                        "effective_batch": stats.final_batch_size,
                        "accumulation": acc,
                        "median_seconds": statistics.median(times),
                        "min_seconds": min(times),
                        "tiles_per_second": stats.tiles / statistics.median(times),
                        "oom_retries": stats.oom_retries,
                    }
                    if device.type == "cuda":
                        entry["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                        entry["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
                    if args.stages:
                        entry["stage_seconds"] = {k: sum(v) for k, v in stage_times.items()}
                    results["runs"].append(entry)
                    print(json.dumps(entry), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

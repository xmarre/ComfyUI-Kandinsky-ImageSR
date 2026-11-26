#!/usr/bin/env python3
"""Real-checkpoint numerical qualification against the official implementation.

Two phases, so the official package (Diffusers, Transformers, ...) never has to be
installed into the ComfyUI environment:

1. ``reference`` - run in an isolated environment with
   ``pip install git+https://github.com/kandinskylab/kandinsky-image-sr``. Executes
   the official ``KandinskyImageSRPipeline`` and stores inputs, per-tile stage
   tensors (encoded latents, transformer velocity, decoded tiles) and outputs.
2. ``native`` - run with the ComfyUI Python environment. Executes this package
   through ComfyUI model management on the same inputs and prints error metrics
   for every stage, the 18 creativity presets and the final images.

Example::

    # official environment
    python scripts/verify_reference.py reference --bundle /models/Kandinsky-ImageSR-2.0 \
        --image photo.png --out ref.pt
    # ComfyUI environment: official bundle, or the single-file checkpoint + FLUX.1 VAE
    python scripts/verify_reference.py native --comfyui /path/to/ComfyUI \
        --bundle /models/Kandinsky-ImageSR-2.0 --reference ref.pt --report report.json
    python scripts/verify_reference.py native --comfyui /path/to/ComfyUI \
        --bundle /models/Kandinsky-ImageSR-2.0 --model /models/kandinsky_imagesr_2.0.safetensors \
        --vae /models/vae/ae.safetensors --reference ref.pt --report report_single.json

Without ``--image`` a deterministic synthetic test pattern is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CREATIVITY = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}
#: Decoded tiles are large; keep only the first few per run for stage comparison.
MAX_DECODED_TILES = 2


def load_image(path: str | None, size: tuple[int, int]) -> torch.Tensor:
    """``(1, 3, H, W)`` in [0, 1]: EXIF-transposed RGB like ComfyUI's LoadImage (no ICC conversion)."""
    if path is None:
        h, w = size
        y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        gen = torch.Generator().manual_seed(0)
        img = torch.stack([x, y, 0.5 + 0.5 * torch.sin(12 * math.pi * x * y)])
        img[:, h // 3 : h // 3 + 3] = 1.0 - img[:, h // 3 : h // 3 + 3]  # thin lines
        img[:, :, w // 2 : w // 2 + 2] = 0.0
        img = img + 0.04 * torch.randn(img.shape, generator=gen)
        return ((img.clamp(0, 1) * 255).round() / 255)[None]
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(rgb.height, rgb.width, 3)
    return (data.float() / 255.0).permute(2, 0, 1)[None]


def runs_from_args(args) -> list[dict]:
    runs = []
    for factor in args.factors:
        for level in args.creativity:
            runs.append({"factor": factor, "creativity": level, "timestep": CREATIVITY[factor][level]})
    return runs


# ----------------------------------------------------------------------------- reference phase


def _cpu_autocast(device: torch.device):
    """On CPU, emulate the official CUDA transformer precision (bf16 autocast).

    The official pipeline enters autocast per stage and only on CUDA: the
    transformer runs under bf16 autocast, VAE encoding runs without autocast and
    VAE decoding under autocast. On CPU it uses no autocast at all, so its
    transformer would compute the modulation projections in FP32. This context is
    applied to the transformer call only. VAE stages are left exactly as the
    official CPU path runs them; CPU autocast cannot reproduce CUDA's FP32 group
    normalisation in the decoder, so decoder precision on CUDA is only qualified by
    running this script on CUDA.
    """
    from contextlib import nullcontext

    return torch.autocast("cpu", dtype=torch.bfloat16) if device.type == "cpu" else nullcontext()


def phase_reference(args) -> None:
    from kandinsky_image_sr import KandinskyImageSRPipeline

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    pipe = KandinskyImageSRPipeline.from_pretrained(args.bundle, torch_dtype=torch.bfloat16).to(device)
    image = load_image(args.image, tuple(args.size))
    record = {"image": image, "runs": [], "presets": {}, "device": str(device), "torch": torch.__version__}
    for run in runs_from_args(args):
        stages = {"lq": [], "pred": [], "decoded": []}
        encode, decode, forward = pipe._encode, pipe._decode, pipe.transformer.forward

        def _encode(pixels, dev, stages=stages, encode=encode):
            out = encode(pixels, dev)
            stages["lq"].append(out.float().cpu())
            return out

        def _decode(latents, stages=stages, decode=decode):
            out = decode(latents)
            if len(stages["decoded"]) < MAX_DECODED_TILES:
                stages["decoded"].append(out.float().cpu())
            return out

        def _forward(*a, stages=stages, forward=forward, **k):
            with _cpu_autocast(device):
                out = forward(*a, **k)
            pred = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
            stages["pred"].append(pred.squeeze(1).permute(0, 3, 1, 2).float().cpu())
            return out

        pipe._encode, pipe._decode, pipe.transformer.forward = _encode, _decode, _forward
        start = time.perf_counter()
        try:
            output = pipe(image[0], upscale_factor=run["factor"], creativity=run["creativity"], output_type="pt").images
        finally:
            pipe._encode, pipe._decode, pipe.transformer.forward = encode, decode, forward
        record["runs"].append({**run, **stages, "output": output.float().cpu(), "seconds": time.perf_counter() - start})
        print(
            f"reference x{run['factor']} creativity {run['creativity']}: {time.perf_counter() - start:.1f}s", flush=True
        )

    first = record["runs"][0]["lq"][0][:1].to(device)
    latent = first.permute(0, 2, 3, 1).unsqueeze(1)
    size = first.shape[-1] // 2
    pos = (torch.arange(1, device=device), torch.arange(size, device=device), torch.arange(size, device=device))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else _cpu_autocast(device)
    for factor, values in CREATIVITY.items():
        for level, t in enumerate(values):
            with autocast:
                out = pipe.transformer(
                    hidden_states=latent.to(torch.bfloat16),
                    encoder_hidden_states=None,
                    timestep=torch.full((1,), t * 1000.0, device=device),
                    pooled_projections=None,
                    visual_rope_pos=pos,
                    text_rope_pos=None,
                    return_dict=False,
                )
            pred = out[0] if isinstance(out, tuple) else out
            record["presets"][f"{factor}:{level}"] = pred.squeeze(1).permute(0, 3, 1, 2).float().cpu()
    record["preset_input"] = first.cpu()
    torch.save(record, args.out)
    print(f"saved {args.out}")


# ----------------------------------------------------------------------------- native phase


def metrics(actual: torch.Tensor, expected: torch.Tensor, data_range: float | None = None) -> dict:
    a, b = actual.double(), expected.double()
    d = a - b
    rms = d.pow(2).mean().sqrt().item()
    out = {
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "rms": rms,
        "rel_rms": rms / max(b.pow(2).mean().sqrt().item(), 1e-30),
        "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }
    if data_range is not None:
        out["psnr_db"] = float("inf") if rms == 0 else 20 * math.log10(data_range / rms)
    return out


def phase_native(args) -> None:
    from _comfy_bootstrap import bootstrap

    bootstrap(args.comfyui, args.comfy_args)
    from kandinsky_sr import runtime
    from kandinsky_sr.pipeline import TiledRestorer

    reference = torch.load(args.reference, weights_only=False)
    torch.set_grad_enabled(False)
    model_path = args.model or args.bundle
    model = runtime.load_model(
        model_path, Path(model_path).name, weight_dtype="bf16", vae_dtype=args.vae_dtype, patcher=args.patcher
    )
    from _comfy_bootstrap import load_flux_vae

    vae = load_flux_vae(args.vae, args.vae_dtype)
    print(
        f"native: {model.bundle.kind} checkpoint, VAE {'from ' + args.vae if vae else 'bundled'}, "
        f"patcher {type(model.patcher).__name__}"
    )
    image = reference["image"].permute(0, 2, 3, 1).contiguous()
    backend_cls = runtime._ComfyBackend
    report = {"reference_device": reference["device"], "runs": [], "presets": {}}
    for ref_run in reference["runs"]:
        for batch in args.tile_batch_sizes:
            stages = {"lq": [], "pred": [], "decoded": []}
            pending: dict[str, torch.Tensor] = {}
            encode, transform, decode = backend_cls.encode, backend_cls.transform, backend_cls.decode
            infer, commit = TiledRestorer._infer, TiledRestorer._commit

            # Stage outputs are held per attempt and recorded only when the pipeline
            # commits the batch, so out-of-memory retries never duplicate tiles.
            def _infer(self, *a, pending=pending, infer=infer):
                pending.clear()
                return infer(self, *a)

            def _commit(self, units, prepared, accumulators, pending=pending, stages=stages, commit=commit):
                commit(self, units, prepared, accumulators)
                stages["lq"].append(pending.pop("lq"))
                stages["pred"].append(pending.pop("pred"))
                decoded = pending.pop("decoded")
                kept = sum(t.shape[0] for t in stages["decoded"])
                if kept < MAX_DECODED_TILES:
                    stages["decoded"].append(decoded[: MAX_DECODED_TILES - kept])

            def _encode(self, pixels, pending=pending, encode=encode):
                out = encode(self, pixels)
                pending["lq"] = out.float().cpu()
                return out

            def _transform(self, latents, index, pending=pending, transform=transform):
                out = transform(self, latents, index)
                pending["pred"] = out.float().cpu()
                return out

            def _decode(self, latents, pending=pending, decode=decode):
                out = decode(self, latents)
                pending["decoded"] = out.float().cpu()
                return out

            backend_cls.encode, backend_cls.transform, backend_cls.decode = _encode, _transform, _decode
            TiledRestorer._infer, TiledRestorer._commit = _infer, _commit
            start = time.perf_counter()
            try:
                output, stats = runtime.upscale(
                    model,
                    image,
                    ref_run["factor"],
                    ref_run["creativity"],
                    tile_batch_size=batch,
                    accumulation_device=args.accumulation_device,
                    vae=vae,
                )
            finally:
                backend_cls.encode, backend_cls.transform, backend_cls.decode = encode, transform, decode
                TiledRestorer._infer, TiledRestorer._commit = infer, commit
            native_lq, native_pred = torch.cat(stages["lq"]), torch.cat(stages["pred"])
            ref_lq, ref_pred = torch.cat(ref_run["lq"]), torch.cat(ref_run["pred"])
            if native_lq.shape != ref_lq.shape or native_pred.shape != ref_pred.shape or stats.tiles != ref_lq.shape[0]:
                raise SystemExit(
                    f"tile mismatch: native {tuple(native_lq.shape)} ({stats.tiles} tiles) vs reference "
                    f"{tuple(ref_lq.shape)}; the inputs or tile geometry differ"
                )
            seconds = time.perf_counter() - start
            output = output.permute(0, 3, 1, 2).float().cpu()
            entry = {
                "factor": ref_run["factor"],
                "creativity": ref_run["creativity"],
                "tile_batch_size": batch,
                "seconds_native": seconds,
                "seconds_reference": ref_run["seconds"],
                "oom_retries": stats.oom_retries,
                "lq": metrics(native_lq, ref_lq),
                "velocity": metrics(native_pred, ref_pred),
                "decoded_tiles": metrics(
                    torch.cat(stages["decoded"])[:MAX_DECODED_TILES],
                    torch.cat(ref_run["decoded"])[:MAX_DECODED_TILES],
                    1.0,
                ),
                "output": metrics(output, ref_run["output"], 1.0),
                "output_8bit_mismatch_fraction": ((output * 255).round() != (ref_run["output"] * 255).round())
                .float()
                .mean()
                .item(),
            }
            report["runs"].append(entry)
            print(
                json.dumps({k: v for k, v in entry.items() if k in ("factor", "creativity", "tile_batch_size")})
                + f" output PSNR {entry['output']['psnr_db']:.2f} dB, velocity rel RMS "
                f"{entry['velocity']['rel_rms']:.2e}, {seconds:.1f}s",
                flush=True,
            )
            if args.error_maps:
                from PIL import Image

                diff = (output - ref_run["output"]).abs().amax(1)[0]
                scale = 255.0 / max(diff.max().item(), 1e-8)
                name = f"error_x{ref_run['factor']}_c{ref_run['creativity']}_b{batch}.png"
                Image.fromarray((diff * scale).clamp(0, 255).byte().numpy()).save(Path(args.error_maps) / name)

    lq = reference["preset_input"].to(model.patcher.load_device)
    import comfy.model_management as mm

    mm.load_models_gpu([model.patcher])
    for key, expected in reference["presets"].items():
        factor, level = (int(v) for v in key.split(":"))
        index = model.config.timestep_index(CREATIVITY[factor][level])
        actual = model.transformer(lq, index).float().cpu()
        report["presets"][key] = {**metrics(actual, expected), "bitwise_equal": bool(torch.equal(actual, expected))}
        print(f"preset x{factor} creativity {level}: rel RMS {report['presets'][key]['rel_rms']:.2e}", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("reference", "native"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True, help="Kandinsky-ImageSR-2.0 bundle directory")
    ref = sub.choices["reference"]
    ref.add_argument("--image", help="input image (default: synthetic pattern)")
    ref.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(600, 700),
        metavar=("H", "W"),
        help="synthetic pattern size (default 600 700: 4 tiles at 2x, 12 at 4x)",
    )
    ref.add_argument("--factors", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    ref.add_argument("--creativity", type=int, nargs="+", default=[0, 4, 8], choices=range(9))
    ref.add_argument("--device", default="cuda")
    ref.add_argument("--out", required=True)
    nat = sub.choices["native"]
    nat.add_argument("--comfyui", required=True, help="ComfyUI checkout")
    nat.add_argument("--reference", required=True, help="file written by the reference phase")
    nat.add_argument("--model", help="single-file checkpoint to test instead of --bundle (same weights)")
    nat.add_argument("--vae", help="FLUX.1 ae.safetensors to use instead of the bundle's VAE")
    nat.add_argument(
        "--vae-dtype",
        default="bf16",
        choices=["default", "bf16", "fp32"],
        help="VAE precision (bf16 matches the reference)",
    )
    nat.add_argument(
        "--patcher",
        default="standard",
        choices=["standard", "dynamic"],
        help="ComfyUI model patcher for the transformer (dynamic needs ComfyUI's dynamic VRAM)",
    )
    nat.add_argument("--tile-batch-sizes", type=int, nargs="+", default=[1, 4])
    nat.add_argument("--accumulation-device", default="auto", choices=["auto", "cpu", "gpu"])
    nat.add_argument("--report", help="write a JSON report")
    nat.add_argument("--error-maps", help="directory for per-run absolute error images")
    nat.add_argument(
        "--comfy-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="ComfyUI flags, e.g. --comfy-args --use-pytorch-cross-attention",
    )
    args = parser.parse_args()
    if args.phase == "reference":
        phase_reference(args)
    else:
        phase_native(args)


if __name__ == "__main__":
    main()

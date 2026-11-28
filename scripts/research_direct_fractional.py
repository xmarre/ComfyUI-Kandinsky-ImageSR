#!/usr/bin/env python3
"""Qualification tools for direct fractional restoration (``restoration_mode="direct_fractional"``).

The geometry, sampling and tile loop are the production ones in
``kandinsky_sr/direct.py`` and ``kandinsky_sr/pipeline.py``; this script only
drives them for experiments. Subcommands::

    tiles       exact transformer tile counts, calibrated vs direct (no model needed)
    experiment  quality study on clean reference images (needs the model)

``experiment`` degrades each reference (antialiased bicubic reduction by the
scale, optional JPEG) and restores it back to the reference size with:

    A  bicubic enlargement (the direct path's own model input)
    B  calibrated: native 2x restoration + the node's final resize
    E  VAE round trip only (correction strength 0)
    D  direct latent correction at each correction strength s:
       decode(z - s * t * v)
    F  direct residual reconstruction at each s:
       clamp(A + decode(z - s * t * v) - decode(z))

The transformer runs once per image and setting; its velocity is reused for
every strength (it does not depend on the strength). References must be at
most one model tile (1024 px) at the target size so that every result is a
single tile; one configuration per run is cross-checked against the
production node path. An interrupted run resumes in the same ``--out`` only
when ``manifest.json`` there (implementation digest, model and VAE identity,
precision, strengths, ComfyUI arguments) matches the current invocation; each
result records the SHA-256 of its reference image. Metrics: PSNR over RGB,
SSIM on luma, PSNR in a central face box, high-frequency energy (mean absolute Laplacian of luma relative to
the reference; > 1 is sharper than the reference), luma contrast (standard
deviation relative to the reference) and mean colour shift.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch
import torch.nn.functional as F

# Appended, not prepended: the repository root must not shadow ComfyUI's own modules
# (this package has a top-level ``nodes.py`` too).
for _path in (str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])):
    if _path not in sys.path:
        sys.path.append(_path)

from kandinsky_sr.config import CREATIVITY_TIMESTEPS  # noqa: E402
from kandinsky_sr.direct import DirectGrid, DirectRestorer, plan_grid  # noqa: E402
from kandinsky_sr.pipeline import TiledRestorer  # noqa: E402
from kandinsky_sr.resize import resize_rgb, scaled_length  # noqa: E402
from kandinsky_sr.tiling import make_grid  # noqa: E402

TILE = 1024


def _tile_counts(args) -> None:
    rows = []
    for size in args.sizes:
        for s in args.scales:
            scale = Decimal(str(s))
            out = (scaled_length(size, scale), scaled_length(size, scale))
            direct = plan_grid(size, size, out, 2 if scale <= 2 else 4, TILE, args.overlap)
            family = 2 if scale <= 2 else 4
            calibrated = make_grid(size, size, family, TILE, args.overlap)
            rows.append(
                {
                    "input": size,
                    "scale": s,
                    "output": out[0],
                    "direct_tiles": direct.num_tiles,
                    "direct_grid": [direct.rows, direct.cols],
                    "calibrated_tiles": calibrated.num_tiles,
                    "calibrated_restored": size * family,
                }
            )
            print(
                f"{size}x{size} -> {out[0]}x{out[1]} ({s}x): direct {direct.num_tiles} tiles "
                f"({direct.rows}x{direct.cols}), calibrated {family}x {calibrated.num_tiles} tiles "
                f"(restores {size * family}x{size * family})"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


# ----------------------------------------------------------------------------- image helpers


def _read(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)[None]


def _write(tensor: torch.Tensor, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((tensor[0].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)).save(path)


def _jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((image[0].numpy() * 255.0 + 0.5).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return torch.from_numpy(np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0)[None]


def _luma(image: torch.Tensor) -> torch.Tensor:
    return (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114)[:, None].double()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR over all RGB channels (peak 1.0)."""
    mse = float(((a.double() - b.double()) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM on luma, 11x11 Gaussian window (sigma 1.5), valid region."""
    x, y = _luma(a), _luma(b)
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    vx, vy, cxy = F.conv2d(x * x, w) - mx**2, F.conv2d(y * y, w) - my**2, F.conv2d(x * y, w) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))).mean())


def _hf_energy(image: torch.Tensor) -> float:
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float64)[None, None]
    return float(F.conv2d(_luma(image), lap).abs().mean())


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    h, w = ref.shape[1:3]
    box = (slice(int(0.3 * h), int(0.8 * h)), slice(int(0.25 * w), int(0.75 * w)))
    return {
        "psnr": round(psnr(out, ref), 3),
        "ssim": round(ssim(out, ref), 4),
        "face_psnr": round(psnr(out[:, box[0], box[1]], ref[:, box[0], box[1]]), 3),
        "hf_ratio": round(_hf_energy(out) / _hf_energy(ref), 3),
        "contrast_ratio": round(float(_luma(out).std() / _luma(ref).std()), 4),
        "color_shift": [round(float(v), 4) for v in (out - ref).mean(dim=(0, 1, 2))],
    }


# ----------------------------------------------------------------------------- experiment


def model_input(backend, grid, images: torch.Tensor) -> torch.Tensor:
    """The single model tile ``(1, 3, T, T)`` the production restorer feeds for ``grid``.

    A :class:`DirectGrid` samples the input directly; the native grid (an exact
    2x request) uses the native tile preprocessing.
    """
    if grid.num_tiles != 1:
        raise ValueError(f"the experiment needs a single model tile, the plan has {grid.num_tiles}")
    restorer = (DirectRestorer if isinstance(grid, DirectGrid) else TiledRestorer)(backend, grid)
    return restorer._model_pixels(images, [(0, 0, 0)], backend.compute_device)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safetensors_identity(path: Path) -> dict:
    """Size and header digest of a safetensors file (the header lists every tensor's dtype, shape and offsets)."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        header = handle.read(int.from_bytes(prefix, "little")) if len(prefix) == 8 else b""
    return {"name": path.name, "bytes": path.stat().st_size, "header_sha256": _sha256(prefix + header)}


def implementation_digest() -> str:
    """Digest of the package sources and this script, which together define every result."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "kandinsky_sr").glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def manifest(args) -> dict:
    """Everything besides the image content that determines a result."""
    return {
        "implementation_sha256": implementation_digest(),
        "torch": torch.__version__,
        "model": _safetensors_identity(Path(args.model)),
        "vae": _safetensors_identity(Path(args.vae)),
        "weight_dtype": "bf16",
        "vae_dtype": args.vae_dtype,
        "family": 2,
        "overlap": 0.25,
        "strengths": [float(s) for s in args.strengths],
        "comfy_args": list(args.comfy_args),
    }


def check_resume(out_dir: Path, current: dict) -> list:
    """Existing results of ``out_dir`` if they were produced under ``current``, else ``SystemExit``.

    Results are only reused when every input that determines them is unchanged;
    anything else must go to a new output directory.
    """
    results_path, manifest_path = out_dir / "results.json", out_dir / "manifest.json"
    if not results_path.exists():
        return []
    if not manifest_path.exists():
        raise SystemExit(f"{out_dir}: results without a manifest cannot be resumed; choose a new --out")
    previous = json.loads(manifest_path.read_text())
    if previous != current:
        changed = sorted(k for k in current.keys() | previous.keys() if current.get(k) != previous.get(k))
        raise SystemExit(f"{out_dir}: existing results were produced with different {', '.join(changed)}; "
                         "choose a new --out")  # fmt: skip
    return json.loads(results_path.read_text())


def _load(args):
    from _comfy_bootstrap import bootstrap, load_flux_vae

    bootstrap(args.comfyui, args.comfy_args)
    import comfy.model_management as mm

    from kandinsky_sr import runtime
    from kandinsky_sr.model import build_rope

    model = runtime.load_model(args.model, Path(args.model).name, weight_dtype="bf16")
    vae = runtime.resolve_vae(model, load_flux_vae(args.vae, args.vae_dtype))
    mm.load_models_gpu([model.patcher, vae.patcher])
    backend = runtime._ComfyBackend(model, vae, build_rope(model.config, model.patcher.load_device))
    return runtime, model, vae, backend


def _experiment(args) -> None:
    out_dir = Path(args.out)
    current = manifest(args)
    results = check_resume(out_dir, current)  # before the model is loaded
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(current, indent=1))
    runtime, model, vae, backend = _load(args)
    done = {(r["image"], r["image_sha256"], r["scale"], r["creativity"], r["jpeg"]) for r in results}
    checked = False
    for path in map(Path, args.images):
        reference = _read(path)
        name = path.stem
        image_sha256 = _sha256(path.read_bytes())
        h, w = reference.shape[1:3]
        for s in args.scales:
            scale = Decimal(str(s))
            lq_h, lq_w = int(Decimal(h) / scale), int(Decimal(w) / scale)
            if scaled_length(lq_h, scale) != h or scaled_length(lq_w, scale) != w or max(h, w) > TILE:
                raise SystemExit(f"{name}: {h}x{w} must be <= {TILE} and reachable from an integer size at {s}x")
            lq = resize_rgb(reference, (lq_h, lq_w)) if (lq_h, lq_w) != (h, w) else reference
            if args.jpeg:
                lq = _jpeg(lq, args.jpeg)
            folder = out_dir / name / f"s{s}_q{args.jpeg}"
            _write(reference, out_dir / name / "reference.png")
            _write(lq, folder / "input.png")
            grid = plan_grid(lq_h, lq_w, (h, w), 2, TILE, 0.25)
            assert isinstance(grid, DirectGrid) or s == 2.0
            for c in args.creativity:
                if (name, image_sha256, s, c, args.jpeg) in done:
                    continue
                t = CREATIVITY_TIMESTEPS[2][c]
                index = model.config.timestep_index(t)
                record = {
                    "image": name,
                    "image_sha256": image_sha256,
                    "scale": s,
                    "creativity": c,
                    "jpeg": args.jpeg,
                    "timestep": t,
                }
                record["input_size"], record["output_size"] = [lq_h, lq_w], [h, w]
                start = time.perf_counter()
                with torch.no_grad():
                    calibrated, stats = runtime.upscale(model, lq, 2, c, output_scale=s, vae=vae)
                record["B_calibrated"] = {
                    **metrics(calibrated, reference),
                    "tiles": stats.tiles,
                    "seconds": round(time.perf_counter() - start, 1),
                }
                _write(calibrated, folder / f"B_calibrated2x_c{c}.png")
                with torch.no_grad():
                    start = time.perf_counter()
                    base = model_input(backend, grid, lq)  # (1, 3, T, T)
                    z = backend.encode(base)
                    v = backend.transform(z, index)
                    seconds_model = time.perf_counter() - start
                    crop = (slice(None), slice(0, h), slice(0, w))
                    a = base.permute(0, 2, 3, 1)[crop].cpu()
                    record["A_bicubic"] = metrics(a, reference)
                    _write(a, folder / "A_bicubic.png")
                    vae_only = backend.decode(z)
                    record["E_vae_only"] = metrics(vae_only.permute(0, 2, 3, 1)[crop].cpu(), reference)
                    _write(vae_only.permute(0, 2, 3, 1)[crop].cpu(), folder / "E_vae_only.png")
                    record["direct"] = {}
                    for strength in args.strengths:
                        coefficient = t if strength == 1.0 else strength * t
                        decoded = backend.decode(z - coefficient * v) if strength else vae_only
                        latent = decoded.permute(0, 2, 3, 1)[crop].cpu()
                        residual = (base + decoded - vae_only).clamp(0, 1).permute(0, 2, 3, 1)[crop].cpu()
                        record["direct"][str(strength)] = {
                            "applied_coefficient": round(strength * t, 6),
                            "D_latent": metrics(latent, reference),
                            "F_residual": metrics(residual, reference),
                        }
                        _write(latent, folder / f"D_direct_c{c}_s{strength}.png")
                        _write(residual, folder / f"F_residual_c{c}_s{strength}.png")
                        if not checked and strength == args.strengths[len(args.strengths) // 2]:
                            # the production node path must produce the same image
                            node, nstats = runtime.upscale(
                                model, lq, 2, c, output_scale=s, vae=vae,
                                restoration_mode="direct_fractional", correction_strength=strength,
                            )  # fmt: skip
                            record["production_check"] = {
                                "strength": strength,
                                "max_abs_diff": float((node - latent).abs().max()),
                                "transformer_evaluations": nstats.transformer_evaluations,
                            }
                            checked = True
                    record["direct_model_seconds"] = round(seconds_model, 1)
                del base, z, v, vae_only
                results.append(record)
                (out_dir / "results.json").write_text(json.dumps(results, indent=1))
                print(json.dumps({k: record[k] for k in ("image", "scale", "creativity", "timestep")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tiles = sub.add_parser("tiles", help="tile counts (no model needed)")
    tiles.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4576])
    tiles.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.75, 2.0])
    tiles.add_argument("--overlap", type=float, default=0.25)
    tiles.add_argument("--json")
    exp = sub.add_parser("experiment", help="quality study (needs the model)")
    exp.add_argument("--comfyui", required=True)
    exp.add_argument("--model", required=True)
    exp.add_argument("--vae", required=True)
    exp.add_argument("--vae-dtype", default="bf16", choices=["default", "bf16", "fp32"])
    exp.add_argument("--images", nargs="+", required=True, help="clean reference images (<= 1024 px)")
    exp.add_argument("--scales", type=float, nargs="+", default=[1.2])
    exp.add_argument("--creativity", type=int, nargs="+", default=[0, 1])
    exp.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    exp.add_argument("--jpeg", type=int, default=75, help="JPEG quality of the degraded input (0 = none)")
    exp.add_argument("--out", required=True)
    exp.add_argument("--comfy-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    {"tiles": _tile_counts, "experiment": _experiment}[args.command](args)


if __name__ == "__main__":
    main()

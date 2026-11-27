#!/usr/bin/env python3
"""Research prototype: direct restoration at a non-native scale (uncalibrated).

This is NOT the production ``output_scale`` feature. Production output scaling
restores at the calibrated native 2x/4x scale and resizes the finished result.
This script instead feeds the transformer 1024x1024 tiles that are bicubic
enlargements of the input by an arbitrary factor ``s`` (for example 1.2), so a
tile covers ``1024 / s`` input pixels and fewer transformer evaluations are
needed. The released model was calibrated for 2x and 4x only; reusing their
baked timesteps at other scales is an experiment, not a supported mode.

Geometry (all in output pixels):

* output size ``round_half_up(input * s)`` per axis; the effective per-axis scale
  is ``out / in``, so every output pixel centre maps to the input coordinate
  ``(o + 0.5) * in / out - 0.5`` (one global mapping, no per-tile drift);
* 1024-pixel tiles at ``tile_positions(max(out, 1024), 1024, int(1024 * (1 -
  overlap)))``, the official tiling rule applied in output space, with the same
  boundary-aware Hann windows and separable normalisation as the native path;
* each tile is sampled from the whole image with bicubic interpolation
  (``grid_sample``, ``align_corners=False``, taps clamped at the image border, the
  equivalent of replicate padding), so there are no tile-local edge effects;
* integer scales with exact native dimensions are delegated to the native
  pipeline unchanged (the control case).

Subcommands::

    tiles      exact transformer tile counts of native 2x/4x vs direct scales
    run        restore one image directly at a scale (needs the model)
    experiment compare bicubic, native 2x + resize, and direct restoration on
               degraded versions of clean reference images (needs the model)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from pathlib import Path

import torch
import torch.nn.functional as F

# Appended, not prepended: the repository root must not shadow ComfyUI's own modules
# (this package has a top-level ``nodes.py`` too).
for _path in (str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])):
    if _path not in sys.path:
        sys.path.append(_path)

from kandinsky_sr.pipeline import RunOptions, TiledRestorer  # noqa: E402
from kandinsky_sr.resize import scaled_length  # noqa: E402
from kandinsky_sr.tiling import hann_profile, make_grid, tile_positions, validate_overlap  # noqa: E402

TILE = 1024


@dataclass(frozen=True)
class DirectGrid:
    """Output-space tile plan for direct restoration of ``height x width`` to ``out_height x out_width``.

    Exposes the attributes :class:`TiledRestorer` reads from a native grid, with
    ``factor = 1`` because ``tops``/``lefts`` are already output coordinates.
    """

    height: int
    width: int
    out_height: int
    out_width: int
    overlap: float
    tile_size: int = TILE
    factor: int = 1

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1 or self.out_height < self.height or self.out_width < self.width:
            raise ValueError("direct restoration needs a non-empty input and an output at least as large")
        validate_overlap(self.overlap)

    @property
    def stride(self) -> int:
        return max(1, int(self.tile_size * (1.0 - self.overlap)))

    @property
    def padded_out_height(self) -> int:
        return max(self.out_height, self.tile_size)

    @property
    def padded_out_width(self) -> int:
        return max(self.out_width, self.tile_size)

    @cached_property
    def tops(self) -> tuple[int, ...]:
        return tuple(tile_positions(self.padded_out_height, self.tile_size, self.stride))

    @cached_property
    def lefts(self) -> tuple[int, ...]:
        return tuple(tile_positions(self.padded_out_width, self.tile_size, self.stride))

    @property
    def rows(self) -> int:
        return len(self.tops)

    @property
    def cols(self) -> int:
        return len(self.lefts)

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols

    def cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def row_profile(self, row: int) -> torch.Tensor:
        return hann_profile(self.tile_size, first=row == 0, last=row == self.rows - 1)

    def col_profile(self, col: int) -> torch.Tensor:
        return hann_profile(self.tile_size, first=col == 0, last=col == self.cols - 1)

    def window(self, row: int, col: int) -> torch.Tensor:
        return self.row_profile(row)[:, None] * self.col_profile(col)[None, :]

    def weight_profiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        wy = torch.zeros(self.padded_out_height)
        wx = torch.zeros(self.padded_out_width)
        for r, top in enumerate(self.tops):
            wy[top : top + self.tile_size] += self.row_profile(r)
        for c, left in enumerate(self.lefts):
            wx[left : left + self.tile_size] += self.col_profile(c)
        return wy, wx

    def accumulation_bytes(self) -> int:
        return 3 * 4 * self.padded_out_height * self.padded_out_width

    def source_coordinates(self, start: int, length: int, out_length: int, in_length: int) -> torch.Tensor:
        """Input pixel coordinates of output pixel centres ``start .. start + length - 1`` (float64)."""
        o = torch.arange(start, start + length, dtype=torch.float64)
        return (o + 0.5) * (in_length / out_length) - 0.5


def plan(height: int, width: int, scale: Decimal, overlap: float = 0.25, tile: int = TILE):
    """Native ``TileGrid`` for exact 2x/4x, otherwise a :class:`DirectGrid`."""
    out_h, out_w = scaled_length(height, scale), scaled_length(width, scale)
    if scale in (2, 4) and (out_h, out_w) == (height * int(scale), width * int(scale)):
        return make_grid(height, width, int(scale), tile, overlap)
    return DirectGrid(height, width, out_h, out_w, overlap, tile_size=tile)


def sample_tiles(images: torch.Tensor, units, grid: DirectGrid, device) -> torch.Tensor:
    """Bicubic samples ``(N, 3, T, T)`` of BHWC ``images`` on the global output-to-input mapping."""
    t = grid.tile_size
    tiles = []
    for b, r, c in units:
        ys = grid.source_coordinates(grid.tops[r], t, grid.out_height, grid.height)
        xs = grid.source_coordinates(grid.lefts[c], t, grid.out_width, grid.width)
        gy = (2.0 * ys + 1.0) / grid.height - 1.0  # align_corners=False normalisation
        gx = (2.0 * xs + 1.0) / grid.width - 1.0
        sample_grid = torch.stack(torch.broadcast_tensors(gx[None, :], gy[:, None]), dim=-1)[None].float()
        source = images[b : b + 1].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous()
        tile = F.grid_sample(source, sample_grid.to(device), mode="bicubic", padding_mode="border", align_corners=False)
        tiles.append(tile.clamp_(0.0, 1.0).contiguous())
    return torch.cat(tiles)


class DirectRestorer(TiledRestorer):
    """The native tile loop with directly sampled 1024-pixel tiles (no per-tile enlargement)."""

    def _infer(self, images, units, opts):
        backend = self.backend
        pixels = sample_tiles(images, units, self.grid, backend.compute_device)
        self._check_interrupt(opts)
        latents = backend.encode(pixels)
        del pixels
        self._check_interrupt(opts)
        velocity = backend.transform(latents, opts.timestep_index)
        corrected = latents - opts.timestep * velocity
        del latents, velocity
        self._check_interrupt(opts)
        decoded = backend.decode(corrected)
        del corrected
        return decoded


def restore(backend, images: torch.Tensor, grid, timestep: float, index: int, tile_batch_size: int = 1):
    """Run ``grid`` (native or direct) with ``backend``; returns ``(output, stats)``."""
    options = RunOptions(
        timestep=timestep,
        timestep_index=index,
        tile_batch_size=tile_batch_size,
        accumulation_device=torch.device("cpu"),
    )
    cls = DirectRestorer if isinstance(grid, DirectGrid) else TiledRestorer
    return cls(backend, grid).run(images, options)


# ----------------------------------------------------------------------------- command line


def _tile_counts(args) -> None:
    rows = []
    for size in args.sizes:
        native2 = make_grid(size, size, 2, TILE, args.overlap).num_tiles
        native4 = make_grid(size, size, 4, TILE, args.overlap).num_tiles
        for s in args.scales:
            scale = Decimal(str(s))
            direct = plan(size, size, scale, args.overlap).num_tiles
            native = native2 if scale <= 2 else native4
            rows.append({"input": size, "scale": s, "direct_tiles": direct, "native_tiles": native})
            print(f"{size}x{size} at {s}x: direct {direct} tiles, native {'2x' if scale <= 2 else '4x'} {native}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


def _load(args):
    from _comfy_bootstrap import bootstrap, load_flux_vae

    bootstrap(args.comfyui, args.comfy_args)
    from kandinsky_sr import runtime
    from kandinsky_sr.model import build_rope

    model = runtime.load_model(args.model, Path(args.model).name, weight_dtype="bf16")
    vae = runtime.resolve_vae(model, load_flux_vae(args.vae, args.vae_dtype))
    import comfy.model_management as mm

    mm.load_models_gpu([model.patcher, vae.patcher])
    rope = build_rope(model.config, model.patcher.load_device)
    return runtime, model, vae, runtime._ComfyBackend(model, vae, rope)


def _read(path: str) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)[None]


def _write(tensor: torch.Tensor, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((tensor[0].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)).save(path)


def _jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((image[0].numpy() * 255.0 + 0.5).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return torch.from_numpy(np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0)[None]


def _luma(image: torch.Tensor) -> torch.Tensor:
    return (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114)[:, None]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM on luma, 11x11 Gaussian window (sigma 1.5), valid region."""
    x, y = _luma(a).double(), _luma(b).double()
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]

    def f(t):
        return F.conv2d(t, w)

    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def _run(args) -> None:
    from kandinsky_sr.config import CREATIVITY_TIMESTEPS

    runtime, model, _vae, backend = _load(args)
    image = _read(args.image)
    scale = Decimal(str(args.scale))
    grid = plan(image.shape[1], image.shape[2], scale, args.overlap)
    t = CREATIVITY_TIMESTEPS[args.timestep_table][args.creativity]
    start = time.perf_counter()
    with torch.no_grad():
        out, stats = restore(backend, image, grid, t, model.config.timestep_index(t))
    print(f"{grid.num_tiles} tiles, t={t}, {time.perf_counter() - start:.1f}s, output {tuple(out.shape[1:3])}")
    _write(out, Path(args.output))


def _experiment(args) -> None:
    from kandinsky_sr.config import CREATIVITY_TIMESTEPS
    from kandinsky_sr.resize import resize_rgb

    runtime, model, vae, backend = _load(args)
    out_dir = Path(args.out)
    results = []
    for path in args.images:
        clean = _read(path)
        name = Path(path).stem
        h, w = clean.shape[1:3]
        _write(clean, out_dir / name / "reference.png")
        for s in args.scales:
            scale = Decimal(str(s))
            lq_h, lq_w = int(Decimal(h) / scale), int(Decimal(w) / scale)
            if scaled_length(lq_h, scale) != h or scaled_length(lq_w, scale) != w:
                raise SystemExit(f"{name}: {h}x{w} is not reachable from an integer size at {s}x; crop it")
            lq = resize_rgb(clean, (lq_h, lq_w)) if (lq_h, lq_w) != (h, w) else clean
            if args.jpeg:
                lq = _jpeg(lq, args.jpeg)
            _write(lq, out_dir / name / f"s{s}_input.png")
            bicubic = F.interpolate(lq.permute(0, 3, 1, 2), size=(h, w), mode="bicubic", align_corners=False)
            bicubic = bicubic.clamp(0, 1).permute(0, 2, 3, 1)
            record = {"image": name, "scale": s, "input": [lq_h, lq_w], "output": [h, w]}
            record["bicubic"] = {"psnr": psnr(bicubic, clean), "ssim": ssim(bicubic, clean)}
            _write(bicubic, out_dir / name / f"s{s}_A_bicubic.png")
            grid = plan(lq_h, lq_w, scale, args.overlap)
            for c in args.creativity:
                t = CREATIVITY_TIMESTEPS[2][c]
                entry = {"t": t}
                start = time.perf_counter()
                with torch.no_grad():
                    native, stats = runtime.upscale(model, lq, 2, c, output_scale=s, vae=vae)
                entry["native_resized"] = {
                    "psnr": psnr(native, clean),
                    "ssim": ssim(native, clean),
                    "tiles": stats.tiles,
                    "seconds": time.perf_counter() - start,
                }
                _write(native, out_dir / name / f"s{s}_B_native2x_c{c}.png")
                if isinstance(grid, DirectGrid):
                    start = time.perf_counter()
                    with torch.no_grad():
                        direct, dstats = restore(backend, lq, grid, t, model.config.timestep_index(t))
                    entry["direct"] = {
                        "psnr": psnr(direct, clean),
                        "ssim": ssim(direct, clean),
                        "tiles": dstats.tiles,
                        "seconds": time.perf_counter() - start,
                    }
                    _write(direct, out_dir / name / f"s{s}_C_direct_c{c}.png")
                record[f"c{c}"] = entry
                print(json.dumps(record), flush=True)
            results.append(record)
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tiles = sub.add_parser("tiles", help="tile counts (no model needed)")
    tiles.add_argument("--sizes", type=int, nargs="+", default=[1000, 1024, 2048])
    tiles.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.75, 2.0])
    tiles.add_argument("--overlap", type=float, default=0.25)
    tiles.add_argument("--json")
    for name in ("run", "experiment"):
        p = sub.add_parser(name)
        p.add_argument("--comfyui", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--vae", required=True)
        p.add_argument("--vae-dtype", default="bf16", choices=["default", "bf16", "fp32"])
        p.add_argument("--overlap", type=float, default=0.25)
        p.add_argument("--comfy-args", nargs=argparse.REMAINDER, default=[])
    run = sub.choices["run"]
    run.add_argument("--image", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--scale", type=float, required=True)
    run.add_argument("--creativity", type=int, default=1)
    run.add_argument("--timestep-table", type=int, default=2, choices=[2, 4], help="baked timesteps to reuse")
    exp = sub.choices["experiment"]
    exp.add_argument("--images", nargs="+", required=True, help="clean reference images")
    exp.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.75, 2.0])
    exp.add_argument("--creativity", type=int, nargs="+", default=[0, 1, 2, 4])
    exp.add_argument("--jpeg", type=int, default=0, help="JPEG quality applied to the degraded input (0 = none)")
    exp.add_argument("--out", required=True)
    args = parser.parse_args()
    {"tiles": _tile_counts, "run": _run, "experiment": _experiment}[args.command](args)


if __name__ == "__main__":
    main()

"""Direct fractional restoration: model tiles placed at the requested output resolution.

The calibrated path restores at the native 2x/4x scale and resizes the result.
Direct restoration instead feeds the unchanged model 1024x1024 tiles that are
bicubic samples of the input at the *requested* scale, so the number of
transformer evaluations follows the requested output size. The model and its
baked conditioning are unchanged; the timestep comes from an explicitly chosen
calibration family (2x or 4x), which was calibrated for that native scale only.

Geometry (output pixels):

* output ``round_half_up(input * scale)`` per axis (:mod:`.resize`); every output
  pixel centre ``o`` maps to the input coordinate ``(o + 0.5) * in / out - 0.5``,
  one global mapping for all tiles;
* 1024-pixel tiles at ``tile_positions(max(out, 1024), 1024, int(1024 * (1 -
  overlap)))`` -- the official tiling rule applied in output space -- with the
  native boundary-aware Hann windows and separable normalisation;
* each tile is a separable bicubic sample (PyTorch's ``A = -0.75`` kernel, taps
  clamped to the image, i.e. replicate padding) of a bounded crop of the input
  that covers the tile's support, so the whole image is never transferred or
  converted per tile, and overlapping tiles see identical samples.

When the requested size is exactly the calibration family's native size the
native :class:`~.tiling.TileGrid` is used instead (see :func:`plan_grid`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import torch

from .pipeline import TiledRestorer
from .tiling import hann_profile, make_grid, tile_positions, validate_overlap

#: Bicubic convolution parameter used by ``torch.nn.functional.interpolate``.
CUBIC_A = -0.75


@dataclass(frozen=True)
class DirectGrid:
    """Output-space tile plan restoring ``height x width`` directly to ``out_height x out_width``.

    Exposes the attributes :class:`~.pipeline.TiledRestorer` reads from a native
    grid; ``factor`` is 1 because ``tops``/``lefts`` are already output offsets.
    """

    height: int
    width: int
    out_height: int
    out_width: int
    overlap: float
    tile_size: int = 1024
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

    def check_coverage(self) -> None:
        wy, wx = self.weight_profiles()
        if not (bool((wy[: self.out_height] > 0).all()) and bool((wx[: self.out_width] > 0).all())):
            raise ValueError(f"tile overlap {self.overlap} leaves output pixels without blend weight")

    def accumulation_bytes(self) -> int:
        return 3 * 4 * self.padded_out_height * self.padded_out_width

    def source_coordinates(self, start: int, length: int, axis: int) -> torch.Tensor:
        """Input coordinates (float64) of output pixel centres ``start .. start + length - 1``.

        ``axis`` 0 = rows, 1 = columns.
        """
        in_length, out_length = (self.height, self.out_height) if axis == 0 else (self.width, self.out_width)
        o = torch.arange(start, start + length, dtype=torch.float64)
        return (o + 0.5) * (in_length / out_length) - 0.5

    def sample_bytes(self) -> int:
        """Per-tile sampling buffers (FP32): the largest source crop and both weight matrices."""
        crop_h = min(self.height, math.ceil(self.tile_size * self.height / self.out_height) + 5)
        crop_w = min(self.width, math.ceil(self.tile_size * self.width / self.out_width) + 5)
        return 4 * (3 * crop_h * crop_w + self.tile_size * (crop_h + crop_w))


def plan_grid(height: int, width: int, out_size: tuple[int, int], family: int, tile_size: int, overlap: float):
    """The native grid when ``out_size`` is exactly ``family`` x the input, otherwise a :class:`DirectGrid`."""
    if out_size == (height * family, width * family):
        return make_grid(height, width, family, tile_size, overlap)
    grid = DirectGrid(height, width, out_size[0], out_size[1], validate_overlap(overlap), tile_size=tile_size)
    grid.check_coverage()
    return grid


def cubic_sampling_matrix(coords: torch.Tensor, length: int) -> tuple[torch.Tensor, int]:
    """Bicubic interpolation matrix ``(len(coords), span)`` and the first source index it covers.

    Row ``i`` holds the four ``A = -0.75`` weights of source coordinate
    ``coords[i]``; taps outside ``[0, length)`` are clamped to the border, as in
    ``F.interpolate(mode="bicubic")``. Weights are computed in float64.
    """
    floor = torch.floor(coords)
    t = coords - floor
    a = CUBIC_A
    w = torch.stack(
        [
            ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
            ((a + 2) * t - (a + 3)) * t * t + 1,
            ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
            ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a,
        ],
        dim=1,
    )
    taps = (floor.long()[:, None] + torch.arange(-1, 3)[None, :]).clamp_(0, length - 1)
    first = int(taps.min())
    span = int(taps.max()) - first + 1
    matrix = torch.zeros(len(coords), span, dtype=torch.float64)
    matrix.scatter_add_(1, taps - first, w)
    return matrix, first


def sample_tiles(images: torch.Tensor, units, grid: DirectGrid, device: torch.device) -> torch.Tensor:
    """Model-space tiles ``(N, 3, T, T)`` FP32 on ``device`` for BHWC ``images`` in [0, 1].

    Only the crop each tile needs is moved to ``device`` and converted; the
    result is clamped to [0, 1] like the native bicubic tiles.
    """
    t = grid.tile_size
    tiles = []
    for image_index, row, col in units:
        my, y0 = cubic_sampling_matrix(grid.source_coordinates(grid.tops[row], t, 0), grid.height)
        mx, x0 = cubic_sampling_matrix(grid.source_coordinates(grid.lefts[col], t, 1), grid.width)
        crop = images[image_index, y0 : y0 + my.shape[1], x0 : x0 + mx.shape[1]]
        crop = crop.to(device=device, dtype=torch.float32).clamp(0.0, 1.0).permute(2, 0, 1)
        my = my.to(device=device, dtype=torch.float32)
        mx = mx.to(device=device, dtype=torch.float32)
        tile = torch.matmul(torch.matmul(my, crop), mx.transpose(0, 1))  # (3, T, T)
        del crop, my, mx
        tiles.append(tile.clamp_(0.0, 1.0)[None])
    return torch.cat(tiles).contiguous()


class DirectRestorer(TiledRestorer):
    """The production tile loop with directly sampled model tiles (no per-tile enlargement)."""

    def _model_pixels(self, images: torch.Tensor, units, device: torch.device) -> torch.Tensor:
        return sample_tiles(images, units, self.grid, device)

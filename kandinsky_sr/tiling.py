"""Tile geometry and boundary-aware Hann blending of the official ImageSR pipeline.

The grid, padding, windows and normalisation reproduce
``kandinsky_image_sr.tiling`` / ``KandinskyImageSRPipeline.__call__`` exactly:

* low-resolution tiles of ``tile_size // factor`` pixels on a stride of
  ``max(1, int(lq_tile * (1 - overlap)))``, with one extra terminal tile aligned
  to the far edge when the stride does not land on it;
* images smaller than one tile are replicate-padded on the bottom/right only and
  the output is cropped back to ``input * factor``;
* every output tile is weighted by a periodic Hann window whose halves touching
  the image border are flattened to 1.

Because each window is the outer product of a row profile (depending only on the
tile row) and a column profile (depending only on the tile column), and the grid
is the Cartesian product of row and column positions, the accumulated weight map
factorises: ``W(y, x) = Wy(y) * Wx(x)``. The pipeline therefore stores only the two
1-D weight sums instead of a full-resolution weight image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import torch

from .config import MAX_OVERLAP, MIN_OVERLAP


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets of tiles along one axis (official algorithm)."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def lq_stride(lq_tile: int, overlap: float) -> int:
    return max(1, int(lq_tile * (1.0 - overlap)))


def hann_profile(length: int, *, first: bool, last: bool) -> torch.Tensor:
    """1-D boundary-aware periodic Hann window in FP32 (official formula)."""
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(length) / length)
    if first:
        w[: length // 2] = 1.0
    if last:
        w[length // 2 :] = 1.0
    return w


def validate_overlap(overlap: float) -> float:
    value = float(overlap)
    if not math.isfinite(value) or not MIN_OVERLAP <= value <= MAX_OVERLAP:
        raise ValueError(f"overlap must lie in [{MIN_OVERLAP}, {MAX_OVERLAP}], got {overlap!r}")
    return value


@dataclass(frozen=True)
class TileGrid:
    """Complete tiling plan for one image size."""

    height: int
    width: int
    factor: int
    tile_size: int
    overlap: float

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError(f"image must be at least 1x1, got {self.height}x{self.width}")
        if self.tile_size % self.factor:
            raise ValueError("tile_size must be divisible by the upscale factor")

    @property
    def lq_tile(self) -> int:
        return self.tile_size // self.factor

    @property
    def stride(self) -> int:
        return lq_stride(self.lq_tile, self.overlap)

    @property
    def padded_height(self) -> int:
        return max(self.height, self.lq_tile)

    @property
    def padded_width(self) -> int:
        return max(self.width, self.lq_tile)

    @property
    def needs_padding(self) -> bool:
        return self.padded_height != self.height or self.padded_width != self.width

    @cached_property
    def tops(self) -> tuple[int, ...]:
        return tuple(tile_positions(self.padded_height, self.lq_tile, self.stride))

    @cached_property
    def lefts(self) -> tuple[int, ...]:
        return tuple(tile_positions(self.padded_width, self.lq_tile, self.stride))

    @property
    def rows(self) -> int:
        return len(self.tops)

    @property
    def cols(self) -> int:
        return len(self.lefts)

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def out_height(self) -> int:
        return self.height * self.factor

    @property
    def out_width(self) -> int:
        return self.width * self.factor

    @property
    def padded_out_height(self) -> int:
        return self.padded_height * self.factor

    @property
    def padded_out_width(self) -> int:
        return self.padded_width * self.factor

    def cells(self) -> list[tuple[int, int]]:
        """Row-major (row, col) tile order, identical to the reference."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def row_profile(self, row: int) -> torch.Tensor:
        return hann_profile(self.tile_size, first=row == 0, last=row == self.rows - 1)

    def col_profile(self, col: int) -> torch.Tensor:
        return hann_profile(self.tile_size, first=col == 0, last=col == self.cols - 1)

    def window(self, row: int, col: int) -> torch.Tensor:
        """2-D window ``(tile, tile)``: the exact reference product ``wy[:, None] * wx[None, :]``."""
        return self.row_profile(row)[:, None] * self.col_profile(col)[None, :]

    def weight_profiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        """1-D accumulated weights ``Wy`` (padded output height) and ``Wx`` (width)."""
        wy = torch.zeros(self.padded_out_height)
        wx = torch.zeros(self.padded_out_width)
        for r, top in enumerate(self.tops):
            wy[top * self.factor : top * self.factor + self.tile_size] += self.row_profile(r)
        for c, left in enumerate(self.lefts):
            wx[left * self.factor : left * self.factor + self.tile_size] += self.col_profile(c)
        return wy, wx

    def check_coverage(self) -> None:
        """Assert every output pixel receives a strictly positive blend weight."""
        wy, wx = self.weight_profiles()
        if not (bool((wy > 0).all()) and bool((wx > 0).all())):
            raise ValueError(
                f"tile overlap {self.overlap} leaves output pixels without blend weight for a "
                f"{self.height}x{self.width} input at x{self.factor}"
            )

    def accumulation_bytes(self) -> int:
        """FP32 RGB accumulator size for one image."""
        return 3 * 4 * self.padded_out_height * self.padded_out_width


def make_grid(height: int, width: int, factor: int, tile_size: int, overlap: float) -> TileGrid:
    grid = TileGrid(height, width, factor, tile_size, validate_overlap(overlap))
    grid.check_coverage()
    return grid

"""Geometry and blending of the direct fractional restoration research prototype."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kandinsky_sr.pipeline import TiledRestorer
from kandinsky_sr.tiling import TileGrid, make_grid
from tests.test_pipeline import TILE, FakeBackend, _images, _opts


def _research():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research_direct_fractional.py"
    spec = importlib.util.spec_from_file_location("research_direct_fractional", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


rd = _research()


class IdentityBackend(FakeBackend):
    """Pixels pass through unchanged (zero velocity): isolates sampling and blending."""

    def encode(self, pixels):
        return pixels * 2 - 1

    def transform(self, latents, timestep_index):
        return torch.zeros_like(latents)

    def decode(self, latents):
        return (latents + 1) * 0.5


def test_integer_scales_use_the_native_grid_unchanged():
    for factor in (2, 4):
        grid = rd.plan(150, 170, Decimal(factor), 0.25, tile=TILE)
        assert isinstance(grid, TileGrid) and grid == make_grid(150, 170, factor, TILE, 0.25)


def test_native_control_reproduces_the_established_pipeline_bitwise():
    images = _images(2, 150, 170)
    grid = rd.plan(150, 170, Decimal(2), 0.25, tile=TILE)
    expected, _ = TiledRestorer(FakeBackend(), make_grid(150, 170, 2, TILE, 0.25)).run(images, _opts(1))
    actual, _ = rd.restore(FakeBackend(), images, grid, 0.15, 5)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("h,w", [(150, 170), (64, 64), (300, 97)])
def test_direct_grid_in_output_space_matches_native_placement_at_2x(h, w):
    """The output-space rule places 2x tiles and windows exactly where the native grid does."""
    native = make_grid(h, w, 2, TILE, 0.25)
    direct = rd.DirectGrid(h, w, 2 * h, 2 * w, 0.25, tile_size=TILE)
    assert direct.tops == tuple(t * 2 for t in native.tops) and direct.lefts == tuple(t * 2 for t in native.lefts)
    for a, b in zip(direct.weight_profiles(), native.weight_profiles()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("h,w", [(1, 1), (7, 3), (100, 100), (150, 170), (333, 129)])
@pytest.mark.parametrize("scale", ["1", "1.2", "1.5", "1.75", "2.5", "3.2"])
@pytest.mark.parametrize("overlap", [0.05, 0.25, 0.5])
def test_every_output_pixel_has_positive_blend_weight(h, w, scale, overlap):
    grid = rd.DirectGrid(h, w, *(rd.scaled_length(v, Decimal(scale)) for v in (h, w)), overlap, tile_size=TILE)
    wy, wx = grid.weight_profiles()
    assert (wy[: grid.out_height] > 0).all() and (wx[: grid.out_width] > 0).all()
    assert grid.tops[-1] + TILE >= grid.out_height and grid.lefts[-1] + TILE >= grid.out_width


def test_pixel_centre_mapping_is_global_and_symmetric():
    grid = rd.DirectGrid(100, 100, 120, 120, 0.25, tile_size=TILE)
    xs = grid.source_coordinates(0, 120, 120, 100)
    torch.testing.assert_close(xs[0] + xs[-1], torch.tensor(99.0, dtype=torch.float64))  # mirror symmetric
    torch.testing.assert_close(xs[1:] - xs[:-1], torch.full((119,), 100 / 120, dtype=torch.float64))
    # a tile's coordinates are a slice of the global ones: no per-tile drift
    torch.testing.assert_close(grid.source_coordinates(37, 50, 120, 100), xs[37:87])
    torch.testing.assert_close(
        rd.DirectGrid(9, 9, 9, 9, 0.25).source_coordinates(0, 9, 9, 9), torch.arange(9.0).double()
    )


@pytest.mark.parametrize("h,w,scale", [(150, 170, "1.2"), (300, 97, "1.5"), (90, 260, "1.75"), (40, 40, "2.5")])
def test_identity_backend_reproduces_one_global_resampling_without_seams(h, w, scale):
    """Overlapping tiles sample the same global mapping, so blending introduces no seams or offsets."""
    images = _images(1, h, w)
    grid = rd.plan(h, w, Decimal(scale), 0.25, tile=TILE)
    assert isinstance(grid, rd.DirectGrid) and grid.num_tiles > 1 or min(h, w) * float(scale) <= TILE
    out, stats = rd.restore(IdentityBackend(), images, grid, 0.15, 5)
    assert out.shape == (1, grid.out_height, grid.out_width, 3) and stats.tiles == grid.num_tiles
    ys = grid.source_coordinates(0, grid.out_height, grid.out_height, h)
    xs = grid.source_coordinates(0, grid.out_width, grid.out_width, w)
    sample = torch.stack(torch.broadcast_tensors(((2 * xs + 1) / w - 1)[None], ((2 * ys + 1) / h - 1)[:, None]), -1)
    expected = F.grid_sample(
        images.permute(0, 3, 1, 2), sample[None].float(), mode="bicubic", padding_mode="border", align_corners=False
    ).clamp(0, 1)
    torch.testing.assert_close(out, expected.permute(0, 2, 3, 1), rtol=0, atol=2e-6)


def test_constant_image_is_reconstructed_exactly_up_to_rounding():
    images = torch.full((1, 150, 170, 3), 0.4)
    grid = rd.plan(150, 170, Decimal("1.2"), 0.25, tile=TILE)
    out, _ = rd.restore(IdentityBackend(), images, grid, 0.15, 5)
    torch.testing.assert_close(out, torch.full_like(out, 0.4), rtol=0, atol=1e-6)


def test_unit_scale_is_the_identity_with_an_identity_backend():
    images = _images(1, 150, 170)
    out, _ = rd.restore(IdentityBackend(), images, rd.plan(150, 170, Decimal(1), 0.25, tile=TILE), 0.15, 5)
    torch.testing.assert_close(out, images, rtol=0, atol=1e-5)  # float32 round-off only


def test_direct_restoration_uses_fewer_tiles_where_the_geometry_allows():
    # exact counts from the implemented placement (1024-pixel tiles, overlap 0.25)
    counts = {s: rd.plan(2048, 2048, Decimal(s)).num_tiles for s in ("1", "1.2", "1.5", "1.75", "2")}
    assert counts == {"1": 9, "1.2": 9, "1.5": 16, "1.75": 25, "2": 25}

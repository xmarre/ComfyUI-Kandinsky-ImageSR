"""Tile geometry and blending versus the upstream tiling code."""

from __future__ import annotations

import itertools

import pytest
import torch

from kandinsky_sr.config import MAX_OVERLAP, MIN_OVERLAP
from kandinsky_sr.tiling import TileGrid, hann_profile, make_grid, validate_overlap
from tests import reference_impl as ref

SIZES = [1, 7, 255, 256, 257, 383, 384, 385, 448, 511, 512, 513, 640, 895, 896, 897, 1023, 1024, 1025, 1531, 2048, 3001]


def _ref_geometry(h, w, factor, overlap=0.25, tile=1024):
    lq = tile // factor
    stride = max(1, int(lq * (1.0 - overlap)))
    ph, pw = max(h, lq), max(w, lq)
    return ref.tile_positions(ph, lq, stride), ref.tile_positions(pw, lq, stride)


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("overlap", [MIN_OVERLAP, 0.1, 0.25, 0.3, 0.5, MAX_OVERLAP])
def test_positions_match_reference(factor, overlap):
    for h, w in itertools.product(SIZES, [1, 300, 1025]):
        grid = make_grid(h, w, factor, 1024, overlap)
        tops, lefts = _ref_geometry(h, w, factor, overlap)
        assert list(grid.tops) == tops and list(grid.lefts) == lefts, (h, w)
        assert grid.out_height == h * factor and grid.out_width == w * factor


def test_reference_geometry_constants():
    g2 = make_grid(1024, 1024, 2, 1024, 0.25)
    assert (g2.lq_tile, g2.stride, g2.rows, g2.cols) == (512, 384, 3, 3)
    g4 = make_grid(1024, 1024, 4, 1024, 0.25)
    assert (g4.lq_tile, g4.stride, g4.rows, g4.cols) == (256, 192, 5, 5)
    # official benchmark tile grids
    assert make_grid(2048, 2048, 4, 1024, 0.25).num_tiles == 121
    assert make_grid(4096, 4096, 2, 1024, 0.25).num_tiles == 121
    assert make_grid(2048, 2048, 2, 1024, 0.25).num_tiles == 25


def test_terminal_tile_is_edge_aligned_with_larger_overlap():
    g = make_grid(600, 600, 2, 1024, 0.25)
    assert g.tops == (0, 88)  # second tile overlaps the first by 424 px instead of 128 px


def test_window_matches_reference():
    g = make_grid(1500, 1100, 2, 1024, 0.25)
    for r, c in g.cells():
        expected = ref.boundary_aware_window(
            1024, 1024, top=r == 0, bottom=r == g.rows - 1, left=c == 0, right=c == g.cols - 1, device="cpu"
        )
        assert torch.equal(g.window(r, c), expected)


def test_single_tile_window_is_flat():
    g = make_grid(100, 100, 2, 1024, 0.25)
    assert g.num_tiles == 1 and bool((g.window(0, 0) == 1).all())


@pytest.mark.parametrize("h,w,factor", [(1500, 1100, 2), (600, 1300, 4), (100, 3001, 2), (897, 897, 4)])
def test_separable_weights_equal_accumulated_reference_weights(h, w, factor):
    g = make_grid(h, w, factor, 1024, 0.25)
    full = torch.zeros(g.padded_out_height, g.padded_out_width)
    for r, c in g.cells():
        top, left = g.tops[r] * factor, g.lefts[c] * factor
        full[top : top + 1024, left : left + 1024] += g.window(r, c)
    wy, wx = g.weight_profiles()
    torch.testing.assert_close(wy[:, None] * wx[None, :], full, rtol=2e-6, atol=0)
    assert bool((full > 0).all())


@pytest.mark.parametrize("overlap", [MIN_OVERLAP, 0.25, MAX_OVERLAP])
def test_every_output_pixel_is_covered(overlap):
    for h, w in itertools.product(SIZES, SIZES[::5]):
        for factor in (2, 4):
            g = make_grid(h, w, factor, 1024, overlap)
            wy, wx = g.weight_profiles()
            assert bool((wy[: g.out_height] > 1e-3).all()) and bool((wx[: g.out_width] > 1e-3).all())


def test_zero_overlap_is_rejected_because_it_leaves_uncovered_pixels():
    grid = TileGrid(1024, 1024, 2, 1024, 0.0)
    with pytest.raises(ValueError, match="without blend weight"):
        grid.check_coverage()
    for bad in (0.0, 0.8, float("nan"), -0.1):
        with pytest.raises(ValueError, match="overlap"):
            validate_overlap(bad)


def test_hann_profile_formula():
    w = hann_profile(8, first=False, last=False)
    assert w[0] == 0 and torch.allclose(w[4], torch.tensor(1.0))
    assert bool((hann_profile(8, first=True, last=False)[:4] == 1).all())
    assert bool((hann_profile(8, first=False, last=True)[4:] == 1).all())


def test_accumulation_bytes():
    g = make_grid(4096, 4096, 2, 1024, 0.25)
    assert g.accumulation_bytes() == 8192 * 8192 * 12

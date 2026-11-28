"""Direct fractional restoration: geometry, sampling, correction strength and residual reconstruction."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest
import torch
import torch.nn.functional as F

from kandinsky_sr import pipeline as pl
from kandinsky_sr.direct import DirectGrid, DirectRestorer, cubic_taps, plan_grid, sample_tiles
from kandinsky_sr.pipeline import TiledRestorer
from kandinsky_sr.resize import scaled_length
from kandinsky_sr.tiling import TileGrid, make_grid
from tests.test_pipeline import TILE, FakeBackend, _images, _LiveAccumulators, _opts


class IdentityBackend(FakeBackend):
    """Pixels pass through unchanged (zero velocity): isolates sampling and blending."""

    def encode(self, pixels):
        return pixels * 2 - 1

    def transform(self, latents, timestep_index):
        self.calls["transform"] += 1
        return torch.zeros_like(latents)

    def decode(self, latents):
        return (latents + 1) * 0.5


def _direct(h, w, scale, overlap=0.25, family=2):
    out = (scaled_length(h, Decimal(scale)), scaled_length(w, Decimal(scale)))
    return plan_grid(h, w, out, family, TILE, overlap)


def _global_bicubic(images, size):
    nchw = images.permute(0, 3, 1, 2).clamp(0, 1)
    return F.interpolate(nchw, size=size, mode="bicubic", align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)


# ----------------------------------------------------------------------------- geometry


def test_exact_family_size_uses_the_native_grid_unchanged():
    for family in (2, 4):
        grid = plan_grid(150, 170, (150 * family, 170 * family), family, TILE, 0.25)
        assert isinstance(grid, TileGrid) and grid == make_grid(150, 170, family, TILE, 0.25)
    # the 2x family at 4x size is not the family's native size: direct geometry
    assert isinstance(plan_grid(150, 170, (600, 680), 2, TILE, 0.25), DirectGrid)


@pytest.mark.parametrize("h,w", [(150, 170), (64, 64), (300, 97)])
def test_output_space_rule_places_2x_tiles_exactly_like_the_native_grid(h, w):
    native = make_grid(h, w, 2, TILE, 0.25)
    direct = DirectGrid(h, w, 2 * h, 2 * w, 0.25, tile_size=TILE)
    assert direct.tops == tuple(t * 2 for t in native.tops) and direct.lefts == tuple(t * 2 for t in native.lefts)
    for a, b in zip(direct.weight_profiles(), native.weight_profiles()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("h,w", [(1, 1), (7, 3), (100, 100), (150, 170), (333, 129)])
@pytest.mark.parametrize("scale", ["1", "1.2", "1.5", "1.75", "2.5", "3.2"])
@pytest.mark.parametrize("overlap", [0.05, 0.25, 0.5])
def test_every_output_pixel_has_positive_blend_weight(h, w, scale, overlap):
    grid = DirectGrid(h, w, scaled_length(h, Decimal(scale)), scaled_length(w, Decimal(scale)), overlap, TILE)
    wy, wx = grid.weight_profiles()
    assert (wy[: grid.out_height] > 0).all() and (wx[: grid.out_width] > 0).all()
    assert grid.tops[-1] + TILE >= grid.out_height and grid.lefts[-1] + TILE >= grid.out_width


def test_pixel_centre_mapping_is_global_and_symmetric():
    grid = DirectGrid(100, 100, 120, 120, 0.25, tile_size=TILE)
    xs = grid.source_coordinates(0, 120, 1)
    torch.testing.assert_close(xs[0] + xs[-1], torch.tensor(99.0, dtype=torch.float64))
    torch.testing.assert_close(xs[1:] - xs[:-1], torch.full((119,), 100 / 120, dtype=torch.float64))
    torch.testing.assert_close(grid.source_coordinates(37, 50, 1), xs[37:87])  # a tile is a slice: no drift
    unit = DirectGrid(9, 9, 9, 9, 0.25).source_coordinates(0, 9, 0)
    torch.testing.assert_close(unit, torch.arange(9.0).double())


@pytest.mark.parametrize(
    "size,scale,tiles,grid",
    [
        (4576, "1", 36, (6, 6)),
        (4576, "1.2", 49, (7, 7)),
        (4576, "1.5", 81, (9, 9)),
        (4576, "1.75", 121, (11, 11)),
        (2048, "1.2", 9, (3, 3)),
        (1024, "1.2", 4, (2, 2)),
    ],
)
def test_tile_counts_follow_the_requested_output_size(size, scale, tiles, grid):
    plan = plan_grid(size, size, (scaled_length(size, Decimal(scale)),) * 2, 2, 1024, 0.25)
    assert isinstance(plan, DirectGrid) and plan.num_tiles == tiles and (plan.rows, plan.cols) == grid
    assert (plan.out_height, plan.out_width) == (scaled_length(size, Decimal(scale)),) * 2
    assert make_grid(size, size, 2, 1024, 0.25).num_tiles == 144 or size != 4576  # calibrated 2x reference


def test_large_example_has_the_requested_size_and_no_native_resolution_buffers():
    grid = plan_grid(4576, 4576, (5491, 5491), 2, 1024, 0.25)
    assert (grid.out_height, grid.out_width) == (5491, 5491)
    assert grid.accumulation_bytes() == 12 * 5491 * 5491  # not 12 * 9152 * 9152


# ----------------------------------------------------------------------------- sampling


def test_cubic_taps_are_partitions_of_unity_clamped_to_the_image():
    coords = torch.linspace(-0.4, 40.3, 500, dtype=torch.float64)
    taps, weights = cubic_taps(coords, 41)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(500, dtype=torch.float64))
    assert taps.min() == 0 and taps.max() == 40 and taps.shape == (500, 4)


def test_overlapping_tiles_sample_identical_values_bitwise():
    images = _images(1, 150, 170)
    grid = DirectGrid(150, 170, 195, 221, 0.5, tile_size=TILE)
    a = sample_tiles(images, [(0, 0, 0)], grid, torch.device("cpu"))[0]
    b = sample_tiles(images, [(0, 0, 1)], grid, torch.device("cpu"))[0]
    shift = grid.lefts[1]
    assert shift < TILE and torch.equal(a[:, :, shift:], b[:, :, : TILE - shift])


@pytest.mark.parametrize(
    "h,w,scale", [(90, 110, "1.2"), (60, 61, "1.5"), (33, 47, "1.75"), (40, 40, "1"), (50, 30, "2")]
)
def test_tile_samples_equal_one_global_bicubic_resize(h, w, scale):
    """Each tile is a crop of F.interpolate(image, size=output, bicubic): the documented global mapping."""
    images = _images(1, h, w)
    out = (scaled_length(h, Decimal(scale)), scaled_length(w, Decimal(scale)))
    grid = DirectGrid(h, w, *out, 0.25, tile_size=TILE)
    expected = _global_bicubic(images, out)
    for row, col in grid.cells():
        tile = sample_tiles(images, [(0, row, col)], grid, torch.device("cpu"))[0].permute(1, 2, 0)
        top, left = grid.tops[row], grid.lefts[col]
        valid = tile[: out[0] - top, : out[1] - left]
        # float32 summation order differs from F.interpolate: ~3e-6 at most
        torch.testing.assert_close(valid, expected[0, top : top + TILE, left : left + TILE], rtol=0, atol=1e-5)


def test_sampling_reads_only_the_tile_support():
    """Pixels outside a tile's bicubic support are never read (only a bounded crop is moved)."""
    images = _images(1, 300, 300)
    grid = DirectGrid(300, 300, 360, 360, 0.25, tile_size=TILE)
    poisoned = images.clone()
    poisoned[:, 120:, :] = float("nan")  # tile (0, 0) covers output rows < 128, i.e. input rows < ~107 + taps
    tile = sample_tiles(poisoned, [(0, 0, 0)], grid, torch.device("cpu"))
    assert torch.isfinite(tile).all()


# ----------------------------------------------------------------------------- restoration


def _run(backend, images, grid, **kw):
    cls = DirectRestorer if isinstance(grid, DirectGrid) else TiledRestorer
    return cls(backend, grid).run(images, _opts(kw.pop("batch", 1), **kw))


@pytest.mark.parametrize("h,w,scale", [(150, 170, "1.2"), (300, 97, "1.5"), (90, 260, "1.75"), (40, 40, "2.5")])
def test_identity_backend_reproduces_the_global_resize_without_seams(h, w, scale):
    images = _images(1, h, w)
    grid = _direct(h, w, scale)
    out, stats = _run(IdentityBackend(), images, grid)
    assert out.shape == (1, grid.out_height, grid.out_width, 3) and stats.tiles == grid.num_tiles
    torch.testing.assert_close(out, _global_bicubic(images, out.shape[1:3]), rtol=0, atol=1e-5)


def test_constant_and_unit_scale_images():
    out, _ = _run(IdentityBackend(), torch.full((1, 150, 170, 3), 0.4), _direct(150, 170, "1.2"))
    torch.testing.assert_close(out, torch.full_like(out, 0.4), rtol=0, atol=1e-6)
    images = _images(1, 150, 170)
    out, _ = _run(IdentityBackend(), images, _direct(150, 170, "1"))
    torch.testing.assert_close(out, images, rtol=0, atol=1e-5)


def test_batches_and_tile_batches_give_the_same_result():
    images = _images(3, 150, 170)
    grid = _direct(150, 170, "1.3")
    single, _ = _run(FakeBackend(), images, grid)
    batched, stats = _run(FakeBackend(), images, grid, batch=5)
    torch.testing.assert_close(batched, single, rtol=0, atol=1e-6)
    assert stats.tiles == 3 * grid.num_tiles


class RecordingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.indices = []

    def transform(self, latents, timestep_index):
        self.indices.append(timestep_index)
        return super().transform(latents, timestep_index)


@pytest.mark.parametrize("native", [True, False])
def test_correction_strength_scales_only_the_applied_update(native):
    images = _images(1, 60, 70)
    grid = make_grid(60, 70, 2, TILE, 0.25) if native else _direct(60, 70, "1.2")
    full, _ = _run(FakeBackend(), images, grid)
    explicit, _ = _run(FakeBackend(), images, grid, correction_strength=1.0)
    assert torch.equal(full, explicit)
    backend = RecordingBackend()
    half, stats = _run(backend, images, grid, correction_strength=0.5)
    assert backend.indices == [5] * grid.num_tiles  # conditioning unchanged
    assert stats.transformer_evaluations == grid.num_tiles
    # the half update equals the full update at half the coefficient (FakeBackend velocity ignores t)
    reference, _ = _run(FakeBackend(), images, grid, t=0.075)
    torch.testing.assert_close(half, reference, rtol=0, atol=0)
    assert not torch.equal(half, full)


def test_zero_strength_skips_the_transformer_and_is_the_vae_round_trip():
    images = _images(1, 60, 70)
    grid = _direct(60, 70, "1.2")
    backend = FakeBackend()
    out, stats = _run(backend, images, grid, correction_strength=0.0)
    assert backend.calls["transform"] == 0 and stats.transformer_evaluations == 0 and stats.tiles == grid.num_tiles
    zero_t, _ = _run(FakeBackend(), images, grid, t=0.0)  # z - 0 * v
    assert torch.equal(out, zero_t)


def test_residual_reconstruction():
    images = _images(1, 60, 70)
    grid = _direct(60, 70, "1.2")
    base, _ = _run(IdentityBackend(), images, grid)
    # zero correction: the decoded difference vanishes, so the sampled input is returned
    # without any VAE or transformer work
    passthrough = FakeBackend()
    zero, zstats = _run(passthrough, images, grid, correction_strength=0.0, residual=True)
    torch.testing.assert_close(zero, base, rtol=0, atol=1e-6)
    assert passthrough.calls["encode"] == passthrough.calls["decode"] == passthrough.calls["transform"] == 0
    assert zstats.tiles == grid.num_tiles and zstats.transformer_evaluations == zstats.transformer_tiles_submitted == 0
    # sign and magnitude: input + decode(z_sr) - decode(z_lq), clamped once after blending
    backend = FakeBackend()
    latent, _ = _run(FakeBackend(), images, grid)
    vae_only, _ = _run(FakeBackend(), images, grid, correction_strength=0.0)
    residual, stats = _run(backend, images, grid, residual=True)
    torch.testing.assert_close(residual, (base + latent - vae_only).clamp(0, 1), rtol=0, atol=1e-5)
    assert backend.calls["decode"] == 2 * grid.num_tiles and stats.transformer_evaluations == grid.num_tiles
    assert stats.transformer_tiles_submitted == grid.num_tiles
    assert residual.min() >= 0 and residual.max() <= 1


def test_sampling_oom_is_retried_transactionally(monkeypatch):
    from kandinsky_sr import direct

    images = _images(2, 150, 170)
    grid = _direct(150, 170, "1.3")
    expected, _ = _run(FakeBackend(), images, grid, batch=4)
    real = direct.sample_tiles
    calls = {"n": 0}

    def flaky(imgs, units, g, device):
        calls["n"] += 1
        if len(units) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (sampling)")
        return real(imgs, units, g, device)

    monkeypatch.setattr(direct, "sample_tiles", flaky)
    tracker = _LiveAccumulators(monkeypatch)
    out, stats = _run(FakeBackend(), images, grid, batch=4)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
    assert stats.oom_retries == 1 and set(stats.batch_sizes) <= {2} and stats.tiles == 2 * grid.num_tiles
    assert tracker.live == 0


def test_transformer_statistics_separate_committed_and_discarded_work(monkeypatch):
    """An OOM after the transformer ran discards that work: it is submitted, not committed."""
    images = _images(2, 150, 170)
    grid = _direct(150, 170, "1.3")
    backend = FakeBackend()
    real_decode = backend.decode

    def flaky_decode(latents):
        if latents.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (decode)")
        return real_decode(latents)

    monkeypatch.setattr(backend, "decode", flaky_decode)
    _, stats = _run(backend, images, grid, batch=4)
    total = 2 * grid.num_tiles
    assert stats.oom_retries == 1 and stats.transformer_evaluations == total
    assert stats.transformer_tiles_submitted == total + 4  # the discarded first batch of four


@pytest.mark.parametrize("stage", ["sample", "encode", "transform", "decode"])
def test_cancellation_in_every_stage_releases_buffers(monkeypatch, stage):
    from kandinsky_sr import direct

    class Cancel(Exception):
        pass

    images = _images(2, 150, 170)
    grid = _direct(150, 170, "1.3")
    backend = FakeBackend()
    tracker = _LiveAccumulators(monkeypatch)
    calls = {"n": 0}

    def maybe_cancel():
        calls["n"] += 1
        if calls["n"] == 3:
            raise Cancel

    if stage == "sample":
        real = direct.sample_tiles
        monkeypatch.setattr(direct, "sample_tiles", lambda *a: (maybe_cancel(), real(*a))[1])
    else:
        real_stage = getattr(backend, stage)
        monkeypatch.setattr(backend, stage, lambda *a: (maybe_cancel(), real_stage(*a))[1])
    with pytest.raises(Cancel):
        DirectRestorer(backend, grid).run(images, _opts(1))
    import gc

    gc.collect()
    assert tracker.live == 0


def test_direct_grid_is_not_mutated_and_restorers_keep_no_state():
    images = _images(1, 60, 70)
    grid = _direct(60, 70, "1.2")
    restorer = DirectRestorer(FakeBackend(), grid)
    first, _ = restorer.run(images, _opts(1, correction_strength=0.3, residual=True))
    plain, _ = restorer.run(images, _opts(1))
    again, _ = DirectRestorer(FakeBackend(), grid).run(images, _opts(1))
    assert torch.equal(plain, again) and not torch.equal(first, plain)
    assert restorer._windows == {}


def test_pipeline_options_default_to_the_official_update():
    opts = _opts(1)
    assert opts.correction_strength == 1.0 and opts.residual is False
    assert math.isclose(pl.RunOptions(0.15, 5, 1, torch.device("cpu")).correction_strength, 1.0)

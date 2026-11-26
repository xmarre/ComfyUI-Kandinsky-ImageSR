"""Tiled pipeline: equivalence with the upstream tile loop, batching and OOM safety."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kandinsky_sr import pipeline as pl
from kandinsky_sr.pipeline import RunOptions, TileBatchOOMError, TiledRestorer
from kandinsky_sr.tiling import make_grid
from tests import reference_impl as ref

TILE = 128  # small tile keeps tests fast; geometry code is size independent
LATENT = TILE // 8


class FakeBackend:
    """Deterministic, batch-independent stand-in for VAE + transformer."""

    compute_device = torch.device("cpu")

    def __init__(self):
        self.calls = {"encode": 0, "transform": 0, "decode": 0}
        self.batch_sizes = []

    def encode(self, pixels):
        self.calls["encode"] += 1
        self.batch_sizes.append(pixels.shape[0])
        assert pixels.dtype == torch.float32 and pixels.shape[-1] == TILE
        # reduced-precision VAE kernels depend on memory layout; the reference feeds NCHW
        assert pixels.is_contiguous(memory_format=torch.contiguous_format)
        assert float(pixels.min()) >= 0 and float(pixels.max()) <= 1
        z = F.avg_pool2d(pixels * 2 - 1, 8)
        return torch.cat([z, z.square(), torch.zeros_like(z).repeat(1, 3, 1, 1)[:, :10]], dim=1)

    def transform(self, latents, timestep_index):
        self.calls["transform"] += 1
        return 0.3 * torch.sin(3 * latents) * (1 + timestep_index / 10)

    def decode(self, latents):
        self.calls["decode"] += 1
        x = F.interpolate(latents[:, :3], scale_factor=8, mode="bilinear", align_corners=False)
        return ((x.clamp(-1, 1) + 1) * 0.5).float()


def _images(batch, h, w, seed=0):
    gen = torch.Generator().manual_seed(seed)
    base = torch.rand(batch, 3, max(1, h // 4), max(1, w // 4), generator=gen)
    img = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
    return (img + 0.05 * torch.randn(batch, 3, h, w, generator=gen)).clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def _opts(batch_size=1, t=0.15, index=5, **kw):
    return RunOptions(
        timestep=t, timestep_index=index, tile_batch_size=batch_size, accumulation_device=torch.device("cpu"), **kw
    )


def _oracle(image_hwc, factor, overlap=0.25, t=0.15, index=5):
    backend = FakeBackend()

    def infer(pixels):
        lq = backend.encode(pixels)
        return backend.decode(lq - t * backend.transform(lq, index))

    out = ref.reference_tiled_call(
        image_hwc.permute(2, 0, 1)[None], infer, upscale_factor=factor, tile_size=TILE, overlap=overlap
    )
    return out[0].permute(1, 2, 0)


@pytest.mark.parametrize(
    "h,w,factor", [(64, 64, 2), (40, 50, 2), (17, 200, 4), (150, 90, 2), (97, 131, 4), (64, 192, 2)]
)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_matches_upstream_tile_loop(h, w, factor, batch_size):
    images = _images(1, h, w)
    grid = make_grid(h, w, factor, TILE, 0.25)
    out, stats = TiledRestorer(FakeBackend(), grid).run(images, _opts(batch_size))
    expected = _oracle(images[0], factor)
    assert out.shape == (1, h * factor, w * factor, 3)
    torch.testing.assert_close(out[0], expected, rtol=0, atol=3e-6)
    assert stats.tiles == grid.num_tiles


@pytest.mark.parametrize("overlap", [0.05, 0.5, 0.75])
def test_matches_upstream_at_overlap_extremes(overlap):
    images = _images(1, 150, 170)
    grid = make_grid(150, 170, 2, TILE, overlap)
    out, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    torch.testing.assert_close(out[0], _oracle(images[0], 2, overlap), rtol=0, atol=3e-6)


def test_batched_equals_unbatched_and_batches_are_used():
    images = _images(2, 150, 170)
    grid = make_grid(150, 170, 2, TILE, 0.25)
    ref_out, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1))
    backend = FakeBackend()
    out, stats = TiledRestorer(backend, grid).run(images, _opts(4))
    assert torch.equal(out, ref_out)
    total = 2 * grid.num_tiles
    assert stats.batch_sizes == [4] * (total // 4) + ([total % 4] if total % 4 else [])
    assert max(backend.batch_sizes) == 4


def test_multi_image_order_and_independence():
    images = _images(3, 70, 90, seed=3)
    grid = make_grid(70, 90, 2, TILE, 0.25)
    out, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(5))  # batches span images
    for i in range(3):
        single, _ = TiledRestorer(FakeBackend(), grid).run(images[i : i + 1], _opts(1))
        assert torch.equal(out[i], single[0])
    assert not torch.equal(out[0], out[1])


def test_input_not_mutated_and_no_rng_dependence():
    images = _images(1, 100, 100)
    before = images.clone()
    grid = make_grid(100, 100, 2, TILE, 0.25)
    torch.manual_seed(1)
    a, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    torch.manual_seed(2)
    b, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    assert torch.equal(images, before)
    assert torch.equal(a, b)
    TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    torch.manual_seed(1)
    TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    # the pipeline consumes no random numbers
    assert torch.equal(torch.random.get_rng_state(), torch.manual_seed(1).get_state())
    assert torch.isfinite(a).all() and float(a.min()) >= 0 and float(a.max()) <= 1


def test_constant_image_is_reconstructed_exactly_by_identity_backend():
    class Identity(FakeBackend):
        def encode(self, pixels):
            return pixels.repeat(1, 6, 1, 1)[:, :16]

        def transform(self, latents, timestep_index):
            return torch.zeros_like(latents)

        def decode(self, latents):
            return latents[:, :3]

    images = torch.full((1, 300, 200, 3), 0.37)
    for factor in (2, 4):
        grid = make_grid(300, 200, factor, TILE, 0.25)
        out, _ = TiledRestorer(Identity(), grid).run(images, _opts(2))
        torch.testing.assert_close(out, torch.full_like(out, 0.37), rtol=0, atol=1e-6)


def test_flow_correction_uses_selected_timestep():
    images = _images(1, 64, 64)
    grid = make_grid(64, 64, 2, TILE, 0.25)
    a, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1, t=0.07, index=0))
    b, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1, t=0.30, index=15))
    torch.testing.assert_close(a[0], _oracle(images[0], 2, t=0.07, index=0), rtol=0, atol=3e-6)
    torch.testing.assert_close(b[0], _oracle(images[0], 2, t=0.30, index=15), rtol=0, atol=3e-6)
    assert not torch.allclose(a, b)


class OOMBackend(FakeBackend):
    """Raises a CUDA OOM in ``stage`` whenever the batch is larger than ``max_batch``."""

    def __init__(self, stage, max_batch):
        super().__init__()
        self.stage, self.max_batch, self.ooms = stage, max_batch, 0

    def _maybe(self, stage, x):
        if stage == self.stage and x.shape[0] > self.max_batch:
            self.ooms += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def encode(self, pixels):
        self._maybe("encode", pixels)
        return super().encode(pixels)

    def transform(self, latents, idx):
        self._maybe("transform", latents)
        return super().transform(latents, idx)

    def decode(self, latents):
        out = super().decode(latents)
        self._maybe("decode", latents)  # fail after the decode allocated its output
        return out


@pytest.mark.parametrize("stage", ["encode", "transform", "decode"])
def test_oom_retry_is_transactional(stage):
    images = _images(2, 150, 170)
    grid = make_grid(150, 170, 2, TILE, 0.25)
    expected, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1))
    backend = OOMBackend(stage, max_batch=2)
    seen = []
    out, stats = TiledRestorer(backend, grid).run(images, _opts(8, on_oom=seen.append))
    assert torch.equal(out, expected)  # no tile accumulated twice or skipped
    assert stats.oom_retries == 2 and seen == [8, 4]
    assert stats.final_batch_size == 2 and max(stats.batch_sizes) == 2
    assert stats.tiles == 2 * grid.num_tiles


def test_oom_at_batch_size_one_raises_actionable_error():
    grid = make_grid(64, 64, 2, TILE, 0.25)
    with pytest.raises(TileBatchOOMError, match="fixed by the model"):
        TiledRestorer(OOMBackend("transform", max_batch=0), grid).run(_images(1, 64, 64), _opts(1))


def test_oom_during_accumulator_allocation_is_retried(monkeypatch):
    images = _images(2, 100, 100)
    grid = make_grid(100, 100, 2, TILE, 0.25)
    expected, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1))
    real = pl._Accumulator.__init__
    state = {"failed": False}

    def flaky(self, g, device):
        if not state["failed"]:
            state["failed"] = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        real(self, g, device)

    monkeypatch.setattr(pl._Accumulator, "__init__", flaky)
    out, stats = TiledRestorer(FakeBackend(), grid).run(images, _opts(4))
    assert torch.equal(out, expected) and stats.oom_retries == 1


def test_non_oom_errors_propagate_unchanged():
    class Broken(FakeBackend):
        def transform(self, latents, idx):
            raise RuntimeError("shape mismatch in user patch")

    grid = make_grid(64, 64, 2, TILE, 0.25)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        TiledRestorer(Broken(), grid).run(_images(1, 64, 64), _opts(4))


def test_interrupt_and_progress():
    images = _images(1, 150, 170)
    grid = make_grid(150, 170, 2, TILE, 0.25)
    progress = []
    _, stats = TiledRestorer(FakeBackend(), grid).run(images, _opts(3, progress=progress.append))
    assert sum(progress) == grid.num_tiles == stats.tiles

    class Interrupted(Exception):
        pass

    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 3:
            raise Interrupted

    with pytest.raises(Interrupted):
        TiledRestorer(FakeBackend(), grid).run(images, _opts(1, interrupt=interrupt))


def test_rejects_mismatched_input():
    grid = make_grid(64, 64, 2, TILE, 0.25)
    with pytest.raises(ValueError, match="expected images"):
        TiledRestorer(FakeBackend(), grid).run(torch.zeros(1, 64, 65, 3), _opts(1))
    with pytest.raises(ValueError, match="tile_batch_size"):
        TiledRestorer(FakeBackend(), grid).run(torch.zeros(1, 64, 64, 3), _opts(0))


def test_input_outside_unit_range_is_clamped_like_reference():
    images = _images(1, 64, 64) * 1.5 - 0.25
    grid = make_grid(64, 64, 2, TILE, 0.25)
    out, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1))
    expected = _oracle(images[0].clamp(0, 1), 2)
    torch.testing.assert_close(out[0], expected, rtol=0, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cpu_and_gpu_accumulation_agree():
    images = _images(2, 150, 170)
    grid = make_grid(150, 170, 2, TILE, 0.25)
    cpu, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(2))
    opts = _opts(2)
    opts.accumulation_device = torch.device("cuda")
    gpu, _ = TiledRestorer(FakeBackend(), grid).run(images, opts)
    torch.testing.assert_close(gpu, cpu, rtol=0, atol=1e-6)

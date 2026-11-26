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


class _LiveAccumulators:
    """Counts live ``_Accumulator`` objects (released by refcounting in CPython)."""

    def __init__(self, monkeypatch, limit=None):
        import weakref

        self.live = self.peak = 0
        self.refs = []
        real = pl._Accumulator.__init__
        tracker = self

        def init(acc, grid, device):
            if limit is not None and tracker.live + 1 > limit:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (accumulator)")
            real(acc, grid, device)
            tracker.live += 1
            tracker.peak = max(tracker.peak, tracker.live)
            tracker.refs.append(weakref.ref(acc))
            weakref.finalize(acc, tracker._release)

        monkeypatch.setattr(pl._Accumulator, "__init__", init)

    def _release(self):
        self.live -= 1


class BudgetBackend(FakeBackend):
    """Encode fails when the batch plus live accumulators exceed ``budget`` units."""

    def __init__(self, tracker, budget):
        super().__init__()
        self.tracker, self.budget = tracker, budget

    def encode(self, pixels):
        if self.tracker.live + pixels.shape[0] > self.budget:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (activations)")
        return super().encode(pixels)


def test_failed_attempt_releases_the_accumulators_it_created(monkeypatch):
    """Accumulators allocated by a failed attempt must not survive into the retry."""
    images = _images(4, 64, 64)  # one tile per image
    grid = make_grid(64, 64, 2, TILE, 0.25)
    expected, _ = TiledRestorer(FakeBackend(), grid).run(images, _opts(1))
    tracker = _LiveAccumulators(monkeypatch, limit=3)  # the 4th accumulator of the first attempt fails
    out, stats = TiledRestorer(BudgetBackend(tracker, budget=4), grid).run(images, _opts(4))
    assert torch.equal(out, expected)
    # With the three orphaned accumulators kept alive, the batch-2 retry would need
    # 3 + 2 > 4 units and fail again; releasing them makes it the only retry.
    assert stats.oom_retries == 1 and stats.batch_sizes == [2, 2]
    assert tracker.live == 0


@pytest.mark.parametrize("h,w,batch,size", [(64, 64, 8, 8), (64, 64, 5, 3), (150, 170, 3, 4), (150, 170, 2, 7)])
def test_live_accumulators_never_exceed_the_budgeted_bound(monkeypatch, h, w, batch, size):
    from kandinsky_sr.memory import max_active_accumulators

    grid = make_grid(h, w, 2, TILE, 0.25)
    tracker = _LiveAccumulators(monkeypatch)
    TiledRestorer(FakeBackend(), grid).run(_images(batch, h, w), _opts(size))
    assert tracker.peak <= max_active_accumulators(size, grid.num_tiles, batch)
    assert tracker.live == 0


def test_buffers_are_released_after_finalisation_and_on_cancellation(monkeypatch):
    images = _images(2, 150, 170)
    grid = make_grid(150, 170, 2, TILE, 0.25)
    tracker = _LiveAccumulators(monkeypatch)
    seen = {"first_dead": None}

    def interrupt():
        # once image 1 has an accumulator, image 0 has been finalised
        if len(tracker.refs) == 2 and seen["first_dead"] is None:
            seen["first_dead"] = tracker.refs[0]() is None

    TiledRestorer(FakeBackend(), grid).run(images, _opts(1, interrupt=interrupt))
    assert seen["first_dead"] is True  # released while image 1 was still processing
    assert tracker.live == 0

    class Cancelled(Exception):
        pass

    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        if calls["n"] > 5:  # 4 checks per batch: cancels inside the second batch
            raise Cancelled

    tracker2 = _LiveAccumulators(monkeypatch)
    with pytest.raises(Cancelled) as info:
        TiledRestorer(FakeBackend(), grid).run(images, _opts(1, interrupt=cancel))
    # the traceback (and so the run() frame) is still alive here
    assert info.value.__traceback__ is not None
    assert tracker2.refs and all(ref() is None for ref in tracker2.refs)
    frames = []
    tb = info.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    run_frame = next(f for f in frames if f.f_code.co_name == "run")
    assert run_frame.f_locals.get("output") is None


def test_images_with_unusual_strides_reach_the_vae_as_nchw():
    """IMAGE tensors made by permuting NCHW data (size-1 batch, odd strides) must not leak their layout."""
    # smaller than one tile: the tile is the whole image, whose channels-last layout
    # would otherwise survive padding and interpolation
    hwc = _images(1, 40, 50)[0].contiguous()
    nchw = hwc.permute(2, 0, 1)[None]  # e.g. an image decoded to HWC and converted to NCHW
    image = nchw.permute(0, 2, 3, 1).contiguous()  # no copy: (1, H, W, 3) with batch stride 3
    assert image.stride(0) == 3
    assert image.movedim(-1, 1).is_contiguous(memory_format=torch.channels_last)
    grid = make_grid(40, 50, 2, TILE, 0.25)
    out, _ = TiledRestorer(FakeBackend(), grid).run(image, _opts(2))  # FakeBackend asserts NCHW input
    expected, _ = TiledRestorer(FakeBackend(), grid).run(image.clone(memory_format=torch.contiguous_format), _opts(2))
    assert torch.equal(out, expected)

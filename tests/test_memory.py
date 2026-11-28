"""Memory policies and OOM classification."""

from __future__ import annotations

import pytest
import torch

from kandinsky_sr.memory import (
    AUTO_BATCH_CAP,
    TileMemoryEstimate,
    auto_tile_batch,
    choose_accumulation_device,
    is_oom,
    max_active_accumulators,
    transformer_tile_bytes,
)

GIB = 1024**3
CPU, GPU = torch.device("cpu"), torch.device("cuda", 0)


def test_auto_batch_respects_free_memory_cap_and_tile_count():
    assert auto_tile_batch(80 * GIB, 1 * GIB, 4 * GIB, 100) == AUTO_BATCH_CAP
    assert auto_tile_batch(12 * GIB, 1 * GIB, 4 * GIB, 100) == 2
    assert auto_tile_batch(3 * GIB, 1 * GIB, 4 * GIB, 100) == 1  # never below one tile
    assert auto_tile_batch(0, 1 * GIB, 4 * GIB, 100) == 1
    assert auto_tile_batch(80 * GIB, 0, 4 * GIB, 3) == 3  # no point batching beyond the work


def test_per_tile_estimate_is_peak_of_sequential_stages():
    e = TileMemoryEstimate(vae_encode=3, vae_decode=5, transformer=4, io=1)
    assert e.per_tile == 6


def test_transformer_estimate_scales_with_dtype():
    bf16 = transformer_tile_bytes(4096, 2560, 10240, 20, torch.bfloat16)
    fp32 = transformer_tile_bytes(4096, 2560, 10240, 20, torch.float32)
    assert 200 * 1024**2 < bf16 < 2 * GIB and fp32 > bf16


def test_accumulation_device_policy():
    acc = 2 * GIB  # total for all simultaneously live accumulators
    assert choose_accumulation_device("cpu", GPU, acc, 80 * GIB) == CPU
    assert choose_accumulation_device("gpu", GPU, acc, 0) == GPU
    assert choose_accumulation_device("auto", GPU, acc, 80 * GIB) == GPU
    assert choose_accumulation_device("auto", GPU, acc, 6 * GIB) == CPU  # 2 GiB > 25% of 6 GiB
    assert choose_accumulation_device("auto", CPU, acc, 0) == CPU
    with pytest.raises(ValueError, match="GPU compute device"):
        choose_accumulation_device("gpu", CPU, acc, 0)
    with pytest.raises(ValueError, match="accumulation_device"):
        choose_accumulation_device("vram", GPU, acc, 0)


def test_cpu_accumulation_is_system_memory_even_with_gpu_intermediates():
    """ComfyUI's intermediate device is the GPU under --gpu-only; 'cpu' must still mean CPU."""
    for mode, free in (("cpu", 80 * GIB), ("auto", 1 * GIB)):
        assert choose_accumulation_device(mode, GPU, 2 * GIB, free) == CPU


@pytest.mark.parametrize(
    "batch,tiles,images,expected",
    [
        (1, 25, 4, 1),
        (8, 1, 8, 8),
        (8, 1, 3, 3),
        (8, 25, 4, 2),
        (25, 25, 4, 2),
        (26, 25, 4, 2),
        (27, 25, 4, 3),
        (4, 3, 9, 2),
        (5, 3, 9, 3),
    ],
)
def test_max_active_accumulators(batch, tiles, images, expected):
    assert max_active_accumulators(batch, tiles, images) == expected


def test_auto_batch_accounts_for_batch_dependent_accumulators():
    # eight single-tile images: each extra tile also needs its own accumulator
    per_tile, acc = 1 * GIB, 1 * GIB
    extra = lambda b: max_active_accumulators(b, 1, 8) * acc  # noqa: E731
    assert auto_tile_batch(10 * GIB, 0, per_tile, 8) == 8
    assert auto_tile_batch(10 * GIB, 0, per_tile, 8, extra_bytes=extra) == 5


def test_oom_classification():
    assert is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert not is_oom(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    assert not is_oom(ValueError("out of memory"))
    assert not is_oom(KeyError("x"))


def test_genuine_cpu_allocation_failure_is_recognised_narrowly():
    from kandinsky_sr.memory import is_host_alloc_failure

    try:
        torch.empty(2**47, dtype=torch.uint8)  # 128 TiB: refused by the CPU allocator
    except RuntimeError as error:
        genuine = error
    else:  # pragma: no cover - a machine that grants this cannot produce the failure
        pytest.skip("the allocator granted 128 TiB")
    assert is_host_alloc_failure(genuine)
    assert is_host_alloc_failure(MemoryError())
    assert is_host_alloc_failure(RuntimeError("[enforce fail] DefaultCPUAllocator: not enough memory: you tried"))
    assert not is_host_alloc_failure(RuntimeError("shape mismatch"))
    assert not is_host_alloc_failure(RuntimeError("can't allocate memory for the cache"))
    assert not is_host_alloc_failure(ValueError("DefaultCPUAllocator: can't allocate memory"))

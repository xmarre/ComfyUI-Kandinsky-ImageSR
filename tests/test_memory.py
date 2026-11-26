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
    acc = 1 * GIB
    assert choose_accumulation_device("cpu", GPU, CPU, acc, 80 * GIB) == CPU
    assert choose_accumulation_device("gpu", GPU, CPU, acc, 0) == GPU
    assert choose_accumulation_device("auto", GPU, CPU, acc, 80 * GIB) == GPU
    assert choose_accumulation_device("auto", GPU, CPU, acc, 6 * GIB) == CPU  # 2 GiB > 25% of 6 GiB
    assert choose_accumulation_device("auto", CPU, CPU, acc, 0) == CPU
    with pytest.raises(ValueError, match="GPU compute device"):
        choose_accumulation_device("gpu", CPU, CPU, acc, 0)
    with pytest.raises(ValueError, match="accumulation_device"):
        choose_accumulation_device("vram", GPU, CPU, acc, 0)


def test_oom_classification():
    assert is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert not is_oom(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    assert not is_oom(ValueError("out of memory"))
    assert not is_oom(KeyError("x"))

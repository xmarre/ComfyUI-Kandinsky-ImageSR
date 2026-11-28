"""Memory estimates, OOM classification and automatic batch/accumulator policies.

The estimates are deliberately conservative upper bounds for one 1024x1024 tile.
They are used to ask ComfyUI's model manager for working memory and to size tile
batches from the memory that is actually free once the models are resident.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Upper bound on automatically selected tile batches. Each tile is already a
#: 4096-token transformer sequence that saturates large GPUs; larger batches
#: mostly add memory pressure, not throughput.
AUTO_BATCH_CAP = 8
#: Fraction of free memory an automatically selected GPU accumulator may use.
GPU_ACCUMULATOR_FREE_FRACTION = 0.25
_GIB = 1024**3


def dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def transformer_tile_bytes(num_tokens: int, model_dim: int, ff_dim: int, num_heads: int, dtype: torch.dtype) -> int:
    """Peak transformer activation bytes for one tile (excluding weights).

    Counts the residual stream, the FP32 normalisation/modulation temporaries,
    the FP32 RoPE expansion of queries and keys, the feed-forward hidden state and
    a 1.5x allowance for allocator fragmentation. Memory-efficient attention
    (SDPA flash/efficient kernels or ComfyUI's split attention) is linear in the
    sequence length and covered by the same allowance.
    """
    s = dtype_size(dtype)
    n, d = num_tokens, model_dim
    residual = n * d * (s + 4 * 4)  # activation + up to four FP32 temporaries
    qkv = n * d * (3 * s + 2 * 4 * 3)  # q/k/v + FP32 norm and 2x2 RoPE expansion for q and k
    ffn = n * ff_dim * s * 2  # in_layer output and GELU output
    return int(1.5 * (residual + qkv + ffn))


def tile_io_bytes(tile_size: int, latent_size: int, channels: int) -> int:
    """Per-tile pipeline buffers: bicubic tile, decoded tile and latents (FP32)."""
    return 4 * (2 * 3 * tile_size * tile_size + 3 * channels * latent_size * latent_size)


@dataclass(frozen=True)
class TileMemoryEstimate:
    vae_encode: int
    vae_decode: int
    transformer: int
    io: int

    @property
    def per_tile(self) -> int:
        """Stages run sequentially, so the peak is the largest stage plus pipeline buffers."""
        return max(self.vae_encode, self.vae_decode, self.transformer) + self.io


def max_active_accumulators(batch_size: int, tiles_per_image: int, images: int) -> int:
    """Largest number of image accumulators alive at once for a given tile batch.

    Tiles are queued image by image and an image's accumulator is released right
    after the batch that completes it. During a batch, the live accumulators are
    those of the images the batch touches: a window of ``batch_size`` consecutive
    queue entries spans at most ``1 + ceil((batch_size - 1) / tiles_per_image)``
    images.
    """
    if batch_size < 1 or tiles_per_image < 1 or images < 1:
        raise ValueError("batch_size, tiles_per_image and images must be positive")
    return min(images, 1 + -(-(batch_size - 1) // tiles_per_image))


def auto_tile_batch(
    free_bytes: int,
    reserved_bytes: int,
    per_tile: int,
    num_tiles: int,
    cap: int = AUTO_BATCH_CAP,
    extra_bytes=None,
) -> int:
    """Largest batch (at most ``cap``) whose working memory fits in the free memory.

    ``extra_bytes(batch)`` adds batch-dependent memory, such as device-resident
    accumulators. Never returns less than 1.
    """
    usable = free_bytes - reserved_bytes
    for batch in range(min(cap, num_tiles), 1, -1):
        needed = batch * per_tile + (extra_bytes(batch) if extra_bytes is not None else 0)
        if needed <= usable:
            return batch
    return 1


def choose_accumulation_device(
    mode: str,
    compute_device: torch.device,
    accumulator_bytes: int,
    free_bytes_after_batch: int,
) -> torch.device:
    """Resolve ``auto``/``cpu``/``gpu`` to a concrete accumulation device.

    ``accumulator_bytes`` is the total for all accumulators that can be alive at
    once. ``cpu`` always means system memory, independent of where ComfyUI keeps
    intermediate results (``--gpu-only`` keeps them on the GPU).
    """
    cpu = torch.device("cpu")
    if mode == "cpu":
        return cpu
    if mode == "gpu":
        if compute_device.type == "cpu":
            raise ValueError("accumulation_device='gpu' requires a GPU compute device")
        return compute_device
    if mode != "auto":
        raise ValueError(f"accumulation_device must be 'auto', 'cpu' or 'gpu', got {mode!r}")
    if compute_device.type == "cpu":
        return cpu
    if accumulator_bytes <= GPU_ACCUMULATOR_FREE_FRACTION * max(0, free_bytes_after_batch):
        return compute_device
    return cpu


def is_oom(error: BaseException) -> bool:
    """True for device out-of-memory errors, using ComfyUI's classifier when present."""
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None
    if mm is not None:
        checker = getattr(mm, "is_oom", None)
        if checker is not None:
            return bool(checker(error))
        oom_type = getattr(mm, "OOM_EXCEPTION", None)
        if isinstance(oom_type, type) and oom_type is not Exception and isinstance(error, oom_type):
            return True
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def is_host_alloc_failure(error: BaseException) -> bool:
    """True for a failed system-memory allocation (PyTorch's CPU allocator or Python).

    PyTorch reports these as a plain ``RuntimeError`` from ``DefaultCPUAllocator``
    ("can't allocate memory" on Linux/macOS, "not enough memory" on Windows), which
    accelerator out-of-memory classifiers do not recognise.
    """
    if isinstance(error, MemoryError):
        return True
    message = str(error)
    return (
        isinstance(error, RuntimeError)
        and "DefaultCPUAllocator" in message
        and ("can't allocate memory" in message or "not enough memory" in message)
    )


def format_bytes(value: float) -> str:
    return f"{value / _GIB:.2f} GiB"

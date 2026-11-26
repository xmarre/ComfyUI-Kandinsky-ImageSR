"""Tiled one-step restoration pipeline.

Per output tile, exactly as the official ``KandinskyImageSRPipeline``:

1. take the low-resolution tile from the (replicate-padded) input;
2. bicubic-resize it to the fixed output tile size and clamp to [0, 1];
3. encode with the FLUX VAE (posterior mode) and multiply by ``scaling_factor``
   (no FLUX shift);
4. evaluate the baked transformer once at the selected timestep;
5. apply the flow correction ``z_sr = z_lq - t * v`` in FP32;
6. divide by ``scaling_factor`` and decode, clamp to [-1, 1] and map to [0, 1];
7. blend into the output with boundary-aware Hann windows, normalise, crop.

Execution details that go beyond the reference are about scheduling only: tiles of
all images in a ComfyUI batch form one deterministic queue that is processed in
tile batches; a batch is committed to the accumulators only after every stage of
it succeeded, so an out-of-memory retry with a smaller batch never accumulates a
tile twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

from .memory import format_bytes, is_oom
from .tiling import TileGrid

logger = logging.getLogger(__name__)

#: Rows normalised and copied per step when finalising an accumulator.
_FINALIZE_ROWS = 128


class SRBackend(Protocol):
    """Device-side model stages used by the pipeline."""

    compute_device: torch.device

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """``(B, 3, T, T)`` FP32 pixels in [0, 1] -> scaled FP32 latents ``(B, 16, S, S)``."""

    def transform(self, latents: torch.Tensor, timestep_index: int) -> torch.Tensor:
        """Scaled FP32 latents -> FP32 flow velocity of the same shape."""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Scaled FP32 latents -> ``(B, 3, T, T)`` FP32 pixels in [0, 1]."""


class TileBatchOOMError(RuntimeError):
    """Raised when a single tile does not fit in memory."""


@dataclass
class RunOptions:
    timestep: float
    timestep_index: int
    tile_batch_size: int
    accumulation_device: torch.device
    output_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    interrupt: Callable[[], None] | None = None
    progress: Callable[[int], None] | None = None
    #: Called with the failed batch size after an OOM; may free memory (e.g. caches).
    on_oom: Callable[[int], None] | None = None


@dataclass
class RunStats:
    tiles: int = 0
    batches: int = 0
    oom_retries: int = 0
    final_batch_size: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class _Accumulator:
    """FP32 ``(3, H, W)`` blend buffer of one image plus its committed-tile count."""

    def __init__(self, grid: TileGrid, device: torch.device) -> None:
        self.buffer = torch.zeros(3, grid.padded_out_height, grid.padded_out_width, device=device)
        self.committed = 0


class TiledRestorer:
    """Runs the tiled pipeline for a batch of equally sized images."""

    def __init__(self, backend: SRBackend, grid: TileGrid) -> None:
        self.backend = backend
        self.grid = grid
        self._windows: dict[tuple[torch.device, bool, bool, bool, bool], torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers

    def _window(self, row: int, col: int, device: torch.device) -> torch.Tensor:
        g = self.grid
        key = (device, row == 0, row == g.rows - 1, col == 0, col == g.cols - 1)
        window = self._windows.get(key)
        if window is None:
            window = g.window(row, col).to(device)
            self._windows[key] = window
        return window

    def _lq_tiles(self, images: torch.Tensor, units: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
        g = self.grid
        size = g.lq_tile
        tiles = []
        for image_index, row, col in units:
            top, left = g.tops[row], g.lefts[col]
            image = images[image_index : image_index + 1]
            # Replicate padding on the bottom/right only touches tiles that cross the
            # image border, so pad per tile instead of materialising a padded image.
            bottom = min(top + size, g.height)
            right = min(left + size, g.width)
            tile = image[:, top:bottom, left:right, :].movedim(-1, 1)
            # NCHW-contiguous like the reference. ComfyUI images are channels-last in
            # memory; that layout would propagate through interpolation into the VAE,
            # where reduced-precision convolutions select different kernels and the
            # latents measurably diverge from the reference.
            # ``Tensor.to(memory_format=...)`` returns views with unusual strides unchanged
            # (e.g. a permuted size-1 batch dimension), so make the copy explicit.
            tile = tile.to(device=device, dtype=torch.float32).contiguous()
            tile = tile.clamp(0.0, 1.0)  # the reference clamps its input to [0, 1]
            pad_h = size - (bottom - top)
            pad_w = size - (right - left)
            if pad_h or pad_w:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="replicate")
            tiles.append(tile)
        return torch.cat(tiles)

    def _infer(self, images: torch.Tensor, units: list[tuple[int, int, int]], opts: RunOptions) -> torch.Tensor:
        backend = self.backend
        device = backend.compute_device
        g = self.grid
        lq = self._lq_tiles(images, units, device)
        pixels = F.interpolate(lq, size=(g.tile_size, g.tile_size), mode="bicubic", align_corners=False)
        del lq
        pixels = pixels.clamp_(0.0, 1.0)
        if not pixels.is_contiguous():
            raise RuntimeError("internal error: VAE input must be NCHW-contiguous")
        self._check_interrupt(opts)
        latents = backend.encode(pixels)
        del pixels
        self._check_interrupt(opts)
        velocity = backend.transform(latents, opts.timestep_index)
        corrected = latents - opts.timestep * velocity
        del latents, velocity
        self._check_interrupt(opts)
        decoded = backend.decode(corrected)
        del corrected
        return decoded

    @staticmethod
    def _check_interrupt(opts: RunOptions) -> None:
        if opts.interrupt is not None:
            opts.interrupt()

    # ------------------------------------------------------------------ main loop

    def run(self, images: torch.Tensor, opts: RunOptions) -> tuple[torch.Tensor, RunStats]:
        """Restore ``images`` ``(B, H, W, 3)`` in [0, 1]; returns ``(B, H*f, W*f, 3)`` FP32."""
        g = self.grid
        if images.ndim != 4 or images.shape[-1] != 3 or tuple(images.shape[1:3]) != (g.height, g.width):
            raise ValueError(f"expected images shaped (B, {g.height}, {g.width}, 3), got {tuple(images.shape)}")
        if opts.tile_batch_size < 1:
            raise ValueError("tile_batch_size must be at least 1")

        batch = images.shape[0]
        output = torch.empty(batch, g.out_height, g.out_width, 3, device=opts.output_device, dtype=torch.float32)
        wy, wx = g.weight_profiles()
        queue = [(b, r, c) for b in range(batch) for (r, c) in g.cells()]
        accumulators: dict[int, _Accumulator] = {}
        stats = RunStats(final_batch_size=opts.tile_batch_size)
        batch_size = opts.tile_batch_size
        position = 0
        acc_device = opts.accumulation_device

        try:
            while position < len(queue):
                self._check_interrupt(opts)
                units = queue[position : position + batch_size]
                prepared = self._prepare(images, units, opts, accumulators, acc_device)
                if prepared is None:
                    stats.oom_retries += 1
                    if opts.on_oom is not None:
                        opts.on_oom(batch_size)
                    if batch_size == 1:
                        raise TileBatchOOMError(
                            "Kandinsky ImageSR ran out of memory while processing a single "
                            f"{g.tile_size}x{g.tile_size} tile. The tile size is fixed by the model and cannot be "
                            "reduced. Free GPU memory (close other applications or unload other models), start "
                            "ComfyUI with --lowvram so model weights can be partially offloaded, or set "
                            "accumulation_device to 'cpu'."
                        )
                    batch_size = max(1, batch_size // 2)
                    logger.warning("Kandinsky ImageSR: out of memory, retrying with tile_batch_size=%d", batch_size)
                    continue

                self._commit(units, prepared, accumulators)
                del prepared
                position += len(units)
                stats.batches += 1
                stats.tiles += len(units)
                stats.batch_sizes.append(len(units))
                stats.final_batch_size = batch_size
                if opts.progress is not None:
                    opts.progress(len(units))
                for image_index in sorted(i for i, acc in accumulators.items() if acc.committed == g.num_tiles):
                    # pop first so the buffer is released as soon as finalisation returns
                    self._finalize(accumulators.pop(image_index), wy, wx, output[image_index])

            if accumulators:
                raise RuntimeError("internal error: unfinished tile accumulators")
        except BaseException:
            # The traceback keeps this frame alive; drop the output batch with it.
            output = None
            raise
        finally:
            # Drop device buffers now, even when an exception propagates to the caller.
            accumulators.clear()
            self._windows.clear()
        return output, stats

    def _prepare(
        self,
        images: torch.Tensor,
        units: list[tuple[int, int, int]],
        opts: RunOptions,
        accumulators: dict[int, _Accumulator],
        acc_device: torch.device,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        """Run every allocating step of a batch; ``None`` on out-of-memory.

        On failure, accumulators created by this attempt are removed again so the
        retry starts from the same memory state as the failed attempt did.
        Accumulators of earlier batches (which hold committed tiles) are kept.
        """
        created: list[int] = []
        try:
            decoded = self._infer(images, units, opts)
            decoded = decoded.to(acc_device)
            windows = [self._window(r, c, acc_device) for _, r, c in units]
            for image_index, _, _ in units:
                if image_index not in accumulators:
                    accumulators[image_index] = _Accumulator(self.grid, acc_device)
                    created.append(image_index)
            return decoded, windows
        except Exception as error:
            if not is_oom(error):
                raise
        # Out of memory: outside the except block, so the failed attempt's frames
        # (and the tensors they reference) are already released.
        for image_index in created:
            del accumulators[image_index]
        return None

    def _commit(
        self,
        units: list[tuple[int, int, int]],
        prepared: tuple[torch.Tensor, list[torch.Tensor]],
        accumulators: dict[int, _Accumulator],
    ) -> None:
        """In-place accumulation only: no allocations, cannot fail midway."""
        g = self.grid
        decoded, windows = prepared
        for index, (image_index, row, col) in enumerate(units):
            acc = accumulators[image_index]
            top = g.tops[row] * g.factor
            left = g.lefts[col] * g.factor
            acc.buffer[:, top : top + g.tile_size, left : left + g.tile_size].addcmul_(decoded[index], windows[index])
            acc.committed += 1

    def _finalize(self, acc: _Accumulator, wy: torch.Tensor, wx: torch.Tensor, out: torch.Tensor) -> None:
        """Normalise by the separable blend weights, crop, and write ``out`` ``(H, W, 3)``.

        Reads the accumulator without modifying it, so a failure part-way can be
        retried safely (the retry moves the buffer to system memory first).
        """
        out_of_memory = False
        try:
            self._finalize_rows(acc.buffer, wy, wx, out)
        except Exception as error:
            if not is_oom(error) or acc.buffer.device.type == "cpu":
                acc.buffer = None  # the traceback keeps ``acc`` alive, not its storage
                raise
            out_of_memory = True
        if out_of_memory:
            logger.warning("Kandinsky ImageSR: out of memory while finalising; finishing in system memory")
            acc.buffer = acc.buffer.cpu()
            self._finalize_rows(acc.buffer, wy, wx, out)

    def _finalize_rows(self, buffer: torch.Tensor, wy: torch.Tensor, wx: torch.Tensor, out: torch.Tensor) -> None:
        g = self.grid
        device = buffer.device
        wx_dev = wx[: g.out_width].to(device)
        for start in range(0, g.out_height, _FINALIZE_ROWS):
            stop = min(start + _FINALIZE_ROWS, g.out_height)
            weight = wy[start:stop].to(device)[:, None] * wx_dev[None, :]
            rows = buffer[:, start:stop, : g.out_width] / weight
            out[start:stop].copy_(rows.permute(1, 2, 0))


def describe_memory(per_tile: int, batch: int, accumulator: int, acc_device: torch.device) -> str:
    return (
        f"tile batch {batch} (~{format_bytes(per_tile)} per tile), "
        f"accumulator {format_bytes(accumulator)} on {acc_device}"
    )

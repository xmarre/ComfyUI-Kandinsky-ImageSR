"""ComfyUI integration: model loading, model-management lifecycle and execution."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as cfg
from .loader import BundleInfo, inspect_checkpoint, load_transformer_state_dict, validate_tables
from .memory import (
    TileMemoryEstimate,
    auto_tile_batch,
    choose_accumulation_device,
    format_bytes,
    is_oom,
    max_active_accumulators,
    tile_io_bytes,
    transformer_tile_bytes,
)
from .model import KandinskyImageSRTransformer, build_rope, comfy_attention, comfy_operations
from .pipeline import RunOptions, RunStats, TiledRestorer
from .resize import resize_rgb, target_size, validate_output_scale
from .tiling import make_grid
from .vae import ImageSRVAE, validate_imagesr_vae

logger = logging.getLogger(__name__)

WEIGHT_DTYPES = ("default", "bf16", "fp32")
VAE_DTYPES = ("default", "bf16", "fp32")
ACCUMULATION_MODES = ("auto", "cpu", "gpu")
_DTYPE_NAMES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class KandinskySRModel:
    """Loaded ImageSR transformer under ComfyUI model management.

    The VAE is not part of this object: a FLUX.1 VAE from ComfyUI's Load VAE node is
    passed to :func:`upscale`. For the official bundle, its bundled VAE is loaded
    lazily on first use when no VAE is connected.
    """

    name: str
    bundle: BundleInfo
    transformer: KandinskyImageSRTransformer
    patcher: object  # comfy.model_patcher.ModelPatcher
    weight_dtype: torch.dtype
    vae_dtype_choice: str = "default"
    _bundled_vae: object = field(default=None, repr=False)

    @property
    def config(self) -> cfg.TransformerConfig:
        return self.bundle.transformer_config

    @property
    def has_bundled_vae(self) -> bool:
        return self.bundle.vae_path is not None

    def bundled_vae(self) -> object:
        """The bundle's own VAE (``comfy.sd.VAE``), loaded on first use."""
        if self.bundle.vae_path is None:
            raise ValueError(
                "this Kandinsky ImageSR checkpoint has no bundled VAE: connect the FLUX.1 VAE "
                "(ae.safetensors from a Load VAE node) to the upscale node's vae input"
            )
        if self._bundled_vae is None:
            import comfy.model_management as mm

            self._bundled_vae = _load_vae(self.bundle, _resolve_vae_dtype(self.vae_dtype_choice, mm.vae_device()))
        return self._bundled_vae

    def __repr__(self) -> str:
        return f"KandinskySRModel({self.name!r}, {self.bundle.kind}, weights={self.weight_dtype})"


def _resolve_weight_dtype(choice: str, device: torch.device, params: int) -> torch.dtype:
    import comfy.model_management as mm

    if choice in _DTYPE_NAMES:
        return _DTYPE_NAMES[choice]
    if choice != "default":
        raise ValueError(f"weight_dtype must be one of {WEIGHT_DTYPES}, got {choice!r}")
    dtype = mm.unet_dtype(device, model_params=params, supported_dtypes=[torch.bfloat16, torch.float32])
    if dtype not in (torch.bfloat16, torch.float32):
        # A global --fp16-unet/--fp8-*-unet flag selected a precision this model is not
        # qualified for; fall back to the best supported one.
        fallback = torch.bfloat16 if mm.should_use_bf16(device, model_params=params) else torch.float32
        logger.warning("Kandinsky ImageSR: global UNet dtype %s is unsupported, using %s", dtype, fallback)
        dtype = fallback
    return dtype


def _resolve_vae_dtype(choice: str, device: torch.device) -> torch.dtype | None:
    if choice in _DTYPE_NAMES:
        return _DTYPE_NAMES[choice]
    if choice != "default":
        raise ValueError(f"vae_dtype must be one of {VAE_DTYPES}, got {choice!r}")
    import comfy.model_management as mm

    dtype = mm.vae_dtype(device, [torch.bfloat16, torch.float32])
    if dtype not in (torch.bfloat16, torch.float32):
        logger.warning("Kandinsky ImageSR: global VAE dtype %s is unsupported, using float32", dtype)
        dtype = torch.float32
    return dtype


def _load_vae(bundle: BundleInfo, dtype: torch.dtype) -> object:
    import comfy.diffusers_convert
    import comfy.model_management as mm
    import comfy.sd
    from safetensors.torch import load_file

    state = load_file(str(bundle.vae_path), device="cpu")
    expected = set(comfy.diffusers_convert.convert_vae_state_dict(dict(state)).keys())
    vae = comfy.sd.VAE(sd=state, dtype=dtype, device=mm.vae_device())
    del state
    model_keys = set(vae.first_stage_model.state_dict().keys())
    missing = sorted(model_keys - expected)
    if missing or vae.latent_channels != cfg.LATENT_CHANNELS:
        raise cfg.BundleError(f"VAE is not a FLUX autoencoder; missing weights: {missing[:5]}")
    validate_imagesr_vae(vae)
    return vae


def load_model(
    path: str | Path,
    name: str,
    weight_dtype: str = "default",
    vae_dtype: str = "default",
    patcher: str = "standard",
) -> KandinskySRModel:
    """Load a single-file checkpoint or an official bundle directory.

    ``vae_dtype`` applies to the bundled VAE of an official bundle (loaded lazily).
    ``patcher="dynamic"`` selects ComfyUI's dynamic-VRAM ``ModelPatcherDynamic``; it
    is not exposed in the nodes and exists for qualification on supported GPUs.
    """
    import comfy.model_management as mm
    import comfy.model_patcher

    bundle = inspect_checkpoint(path)
    config = bundle.transformer_config
    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    params = sum(int(torch.Size(info.shape).numel()) for info in bundle.tensors.values())
    dtype = _resolve_weight_dtype(weight_dtype, load_device, params)
    if vae_dtype not in VAE_DTYPES:
        raise ValueError(f"vae_dtype must be one of {VAE_DTYPES}, got {vae_dtype!r}")

    with torch.device("meta"):
        model = KandinskyImageSRTransformer(
            config, dtype=dtype, operations=comfy_operations(), attention=comfy_attention
        )
    # In BF16 mode the reference computes the modulation projections from BF16 copies
    # of their FP32 weights (autocast rounds weight and bias on every call). Storing
    # that same rounding once gives identical results and saves ~1.2 GB; FP32 mode
    # keeps them in FP32. The baked tables always stay FP32.
    state = load_transformer_state_dict(bundle, dtype, modulation_dtype=dtype)
    validate_tables(state, config)
    result = model.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:  # strict=True raises; kept as a guard
        raise cfg.BundleError(f"transformer weight mismatch: {result}")
    del state
    model.requires_grad_(False)
    model.eval()
    leftovers = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftovers:
        raise cfg.BundleError(f"transformer tensors were not loaded: {leftovers[:5]}")
    if hasattr(mm, "archive_model_dtypes"):
        mm.archive_model_dtypes(model)
    if patcher == "standard":
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    elif patcher == "dynamic":
        dynamic = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)
        try:
            import comfy.memory_management as memory_management
        except ImportError:  # ComfyUI releases without dynamic VRAM
            memory_management = None
        if dynamic is None or not getattr(memory_management, "aimdo_enabled", False):
            raise RuntimeError("dynamic VRAM is not available in this ComfyUI installation")
        model_patcher = dynamic(model, load_device=load_device, offload_device=offload_device)
    else:
        raise ValueError(f"patcher must be 'standard' or 'dynamic', got {patcher!r}")

    logger.info("Kandinsky ImageSR loaded %s (%s): transformer %s on %s", name, bundle.kind, dtype, load_device)
    return KandinskySRModel(name, bundle, model, model_patcher, dtype, vae_dtype)


def resolve_vae(model: KandinskySRModel, vae: object | None) -> object:
    """The VAE to use: a connected one (validated) or the bundle's own."""
    if vae is None:
        return model.bundled_vae()
    validate_imagesr_vae(vae)
    return vae


class _ComfyBackend:
    def __init__(self, model: KandinskySRModel, vae: object, rope: torch.Tensor) -> None:
        self.model = model
        self.compute_device = model.patcher.load_device
        self.rope = rope
        # Reads the shared VAE's model, device and dtype; never changes them.
        self.vae = ImageSRVAE(vae.first_stage_model, model.bundle.vae_scaling_factor, vae.device, vae.vae_dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels)

    def transform(self, latents: torch.Tensor, timestep_index: int) -> torch.Tensor:
        return self.model.transformer(latents.to(self.compute_device), timestep_index, rope=self.rope)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)


def estimate_tile_memory(model: KandinskySRModel, vae: object) -> TileMemoryEstimate:
    c = model.config
    t, s = c.tile_size, c.baked_latent_size
    return TileMemoryEstimate(
        vae_encode=int(vae.memory_used_encode((1, 3, t, t), vae.vae_dtype)),
        vae_decode=int(vae.memory_used_decode((1, c.out_visual_dim, s, s), vae.vae_dtype)),
        transformer=transformer_tile_bytes(c.num_tokens, c.model_dim, c.ff_dim, c.num_heads, model.weight_dtype),
        io=tile_io_bytes(t, s, c.in_visual_dim),
    )


def validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor shaped (batch, height, width, channels)")
    if image.shape[-1] not in (3, 4):
        raise ValueError(f"image must have 3 (RGB) or 4 (RGBA) channels, got {image.shape[-1]}")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"image is empty: {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image contains NaN or infinite values")


def upscale(
    model: KandinskySRModel,
    image: torch.Tensor,
    upscale_factor: int,
    creativity: int,
    overlap: float = cfg.DEFAULT_OVERLAP,
    tile_batch_size: int = 0,
    accumulation_device: str = "auto",
    vae: object | None = None,
    output_scale: float = 0.0,
) -> tuple[torch.Tensor, RunStats]:
    """Super-resolve a ComfyUI IMAGE batch. ``tile_batch_size=0`` selects automatically.

    ``vae`` is a FLUX.1 ``comfy.sd.VAE`` (e.g. from Load VAE); without it the
    official bundle's own VAE is used. ``upscale_factor`` is the native
    restoration scale (2 or 4). ``output_scale`` 0 returns the native size;
    otherwise the restored image is resized to ``round_half_up(input *
    output_scale)`` (see :mod:`.resize`), with ``1 <= output_scale <=
    upscale_factor``.
    """
    import comfy.model_management as mm
    import comfy.utils

    validate_image(image)
    timestep = cfg.resolve_timestep(upscale_factor, creativity)
    index = model.config.timestep_index(timestep)
    scale = validate_output_scale(output_scale, upscale_factor)
    if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 0:
        raise ValueError(f"tile_batch_size must be 0 (auto) or a positive integer, got {tile_batch_size!r}")
    if accumulation_device not in ACCUMULATION_MODES:
        raise ValueError(f"accumulation_device must be one of {ACCUMULATION_MODES}, got {accumulation_device!r}")

    batch, height, width, channels = image.shape
    alpha = image[..., 3:4] if channels == 4 else None
    rgb = image[..., :3]
    grid = make_grid(height, width, upscale_factor, model.config.tile_size, overlap)
    total_tiles = batch * grid.num_tiles
    final_size = target_size(height, width, scale, upscale_factor)
    # Exactly the native size (also when a fractional scale rounds to it): no resize at all.
    resize = final_size != (grid.out_height, grid.out_width)

    vae = resolve_vae(model, vae)
    estimate = estimate_tile_memory(model, vae)
    per_tile = estimate.per_tile
    acc_bytes = grid.accumulation_bytes()

    def live_accumulator_bytes(tiles: int) -> int:
        return max_active_accumulators(tiles, grid.num_tiles, batch) * acc_bytes

    device = model.patcher.load_device
    intermediate = mm.intermediate_device()
    # The FP32 RGB output lives on ComfyUI's intermediate device for the whole run
    # (on the GPU with --gpu-only), next to the tile working memory.
    on_device = device.type != "cpu" and _same_device(intermediate, device)
    output_bytes = output_buffer_bytes(batch, grid) if on_device else 0
    # The final resize runs after the tile working memory is released, next to the
    # native output: reserve whichever phase needs more.
    resize_bytes = resize_working_bytes(batch, grid, final_size, channels) if on_device and resize else 0

    # Reserve working memory only for tiles that can actually run in one batch.
    requested = min(tile_batch_size, total_tiles) if tile_batch_size > 0 else 1
    gpu_acc = live_accumulator_bytes(requested) if accumulation_device == "gpu" else 0
    working = max(requested * per_tile + gpu_acc, resize_bytes)
    mm.load_models_gpu([model.patcher, vae.patcher], memory_required=working + output_bytes)

    reserved = mm.extra_reserved_memory() if hasattr(mm, "extra_reserved_memory") else 0
    free = mm.get_free_memory(device) - output_bytes if device.type != "cpu" else 0
    if tile_batch_size > 0:
        batch_size = requested
    elif device.type == "cpu":
        batch_size = 1
    else:
        extra = live_accumulator_bytes if accumulation_device == "gpu" else None
        batch_size = auto_tile_batch(free, reserved, per_tile, total_tiles, extra_bytes=extra)
    acc_device = choose_accumulation_device(
        accumulation_device, device, live_accumulator_bytes(batch_size), free - reserved - batch_size * per_tile
    )
    if resize:
        final = f"final output {final_size[1]}x{final_size[0]} ({scale.normalize()}x, antialiased bicubic resize)"
    else:
        final = "final output at native size"
    logger.info(
        "Kandinsky ImageSR: %d image(s) %dx%d -> %dx%d native restoration (%dx, creativity %d, t=%s), %s; "
        "%d tiles (%dx%d grid), tile batch %d (~%s per tile), accumulator %s on %s",
        batch,
        width,
        height,
        grid.out_width,
        grid.out_height,
        upscale_factor,
        creativity,
        timestep,
        final,
        total_tiles,
        grid.cols,
        grid.rows,
        batch_size,
        format_bytes(per_tile),
        format_bytes(acc_bytes),
        acc_device,
    )

    pbar = comfy.utils.ProgressBar(total_tiles)
    options = RunOptions(
        timestep=timestep,
        timestep_index=index,
        tile_batch_size=batch_size,
        accumulation_device=acc_device,
        output_device=intermediate,
        interrupt=mm.throw_exception_if_processing_interrupted,
        progress=pbar.update,
        on_oom=lambda _size: mm.soft_empty_cache(),
    )
    rope = build_rope(model.config, device)
    try:
        with torch.no_grad():
            output, stats = TiledRestorer(_ComfyBackend(model, vae, rope), grid).run(rgb, options)
    finally:
        del rope

    out_dtype = mm.intermediate_dtype() if hasattr(mm, "intermediate_dtype") else torch.float32
    stats.output_size = final_size
    if resize:
        mm.throw_exception_if_processing_interrupted()

        def assemble(native: torch.Tensor) -> torch.Tensor:
            return _finish_resized(native, alpha, final_size, out_dtype, mm.throw_exception_if_processing_interrupted)

    else:

        def assemble(native: torch.Tensor) -> torch.Tensor:
            return _finish_output(native, alpha, grid, out_dtype)

    return assemble_output(output, assemble, final_size, on_oom=mm.soft_empty_cache), stats


def _same_device(a: torch.device, b: torch.device) -> bool:
    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.type == "cpu":
        return True
    current = torch.cuda.current_device() if a.type == "cuda" and torch.cuda.is_available() else 0
    return (current if a.index is None else a.index) == (current if b.index is None else b.index)


def output_buffer_bytes(batch: int, grid) -> int:
    """FP32 RGB output batch."""
    return batch * grid.out_height * grid.out_width * 3 * 4


class OutputAssemblyError(RuntimeError):
    """The restored image could not be assembled into the final output (out of memory)."""


def assemble_output(output: torch.Tensor, assemble, size: tuple[int, int], on_oom=None) -> torch.Tensor:
    """``assemble(output)``; on out-of-memory on an accelerator, again in system memory.

    The native restoration is never rerun: only the assembly step (alpha, final
    resize, dtype conversion) is retried, from the already restored ``output``.
    Errors other than out-of-memory propagate unchanged.
    """
    failure = None
    try:
        return assemble(output)
    except Exception as error:
        if not is_oom(error):
            raise
        traceback.clear_frames(error.__traceback__)
        failure = error if output.device.type == "cpu" else None
    if failure is None:  # outside the except block, so the failed attempt's tensors are released
        logger.warning("Kandinsky ImageSR: out of memory while assembling the output; finishing in system memory")
        if on_oom is not None:
            on_oom()
        output = output.cpu()
        try:
            return assemble(output)
        except Exception as error:
            if not is_oom(error):
                raise
            traceback.clear_frames(error.__traceback__)
            failure = error
    height, width = size
    raise OutputAssemblyError(
        f"Kandinsky ImageSR: the restoration finished, but there is not enough system memory to assemble the "
        f"{width}x{height} output of {output.shape[0]} image(s). Free system memory, process fewer images at once, "
        "or choose a smaller output_scale."
    ) from failure


def resize_working_bytes(batch: int, grid, size: tuple[int, int], channels: int) -> int:
    """Device memory of the final resize beyond the native output (FP32 bound).

    The final batch, plus for the image being resized: its contiguous NCHW copy,
    the separable filter's intermediate and the resized result.
    """
    height, width = size
    final = batch * height * width * channels * 4
    per_image = 3 * 4 * (grid.out_height * grid.out_width + grid.out_height * width + height * width)
    return final + per_image


def _finish_resized(
    output: torch.Tensor,
    alpha: torch.Tensor | None,
    size: tuple[int, int],
    out_dtype: torch.dtype,
    interrupt,
) -> torch.Tensor:
    """Resize the native RGB output to ``size`` one image at a time and append alpha.

    Alpha is resized directly from the original image to ``size`` (bilinear, as
    on the native path); it never passes through the native resolution.
    """
    channels = 3 if alpha is None else 4
    final = torch.empty(output.shape[0], *size, channels, dtype=out_dtype, device=output.device)
    for index in range(output.shape[0]):
        interrupt()
        final[index, ..., :3].copy_(resize_rgb(output[index : index + 1], size)[0])
    if alpha is not None:
        resized = F.interpolate(alpha.movedim(-1, 1).float(), size=size, mode="bilinear", align_corners=False)
        final[..., 3:].copy_(resized.clamp_(0.0, 1.0).movedim(1, -1))
        del resized
    return final


def _finish_output(output: torch.Tensor, alpha: torch.Tensor | None, grid, out_dtype: torch.dtype) -> torch.Tensor:
    if alpha is not None:
        # Alpha is not part of the generative model: resize it bilinearly to the
        # exact output size and append it unchanged in meaning.
        resized = F.interpolate(
            alpha.movedim(-1, 1).float(), size=(grid.out_height, grid.out_width), mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)
        output = torch.cat([output, resized.movedim(1, -1).to(output)], dim=-1)
        del resized
    if output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output

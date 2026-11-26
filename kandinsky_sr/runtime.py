"""ComfyUI integration: model loading, model-management lifecycle and execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as cfg
from .loader import BundleInfo, inspect_bundle, load_transformer_state_dict, validate_tables
from .memory import (
    TileMemoryEstimate,
    auto_tile_batch,
    choose_accumulation_device,
    format_bytes,
    tile_io_bytes,
    transformer_tile_bytes,
)
from .model import KandinskyImageSRTransformer, build_rope, comfy_attention, comfy_operations
from .pipeline import RunOptions, RunStats, TiledRestorer
from .tiling import make_grid
from .vae import ImageSRVAE

logger = logging.getLogger(__name__)

WEIGHT_DTYPES = ("default", "bf16", "fp32")
VAE_DTYPES = ("default", "bf16", "fp32")
ACCUMULATION_MODES = ("auto", "cpu", "gpu")
_DTYPE_NAMES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class KandinskySRModel:
    """Loaded ImageSR bundle: transformer and VAE under ComfyUI model management."""

    name: str
    bundle: BundleInfo
    transformer: KandinskyImageSRTransformer
    patcher: object  # comfy.model_patcher.ModelPatcher
    vae: object  # comfy.sd.VAE
    weight_dtype: torch.dtype

    @property
    def config(self) -> cfg.TransformerConfig:
        return self.bundle.transformer_config

    @property
    def vae_dtype(self) -> torch.dtype:
        return self.vae.vae_dtype

    def __repr__(self) -> str:
        return f"KandinskySRModel({self.name!r}, weights={self.weight_dtype}, vae={self.vae_dtype})"


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
    return vae


def load_model(
    path: str | Path, name: str, weight_dtype: str = "default", vae_dtype: str = "default"
) -> KandinskySRModel:
    import comfy.model_management as mm
    import comfy.model_patcher

    bundle = inspect_bundle(path)
    config = bundle.transformer_config
    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    params = sum(int(torch.Size(info.shape).numel()) for info in bundle.tensors.values())
    dtype = _resolve_weight_dtype(weight_dtype, load_device, params)

    with torch.device("meta"):
        model = KandinskyImageSRTransformer(
            config, dtype=dtype, operations=comfy_operations(), attention=comfy_attention
        )
    state = load_transformer_state_dict(bundle, dtype)
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
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

    vae = _load_vae(bundle, _resolve_vae_dtype(vae_dtype, mm.vae_device()))
    logger.info(
        "Kandinsky ImageSR loaded %s: transformer %s on %s, VAE %s on %s",
        name,
        dtype,
        load_device,
        vae.vae_dtype,
        vae.device,
    )
    return KandinskySRModel(name, bundle, model, patcher, vae, dtype)


class _ComfyBackend:
    def __init__(self, model: KandinskySRModel, rope: torch.Tensor) -> None:
        self.model = model
        self.compute_device = model.patcher.load_device
        self.rope = rope
        vae = model.vae
        self.vae = ImageSRVAE(vae.first_stage_model, model.bundle.vae_scaling_factor, vae.device, vae.vae_dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels)

    def transform(self, latents: torch.Tensor, timestep_index: int) -> torch.Tensor:
        return self.model.transformer(latents.to(self.compute_device), timestep_index, rope=self.rope)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)


def estimate_tile_memory(model: KandinskySRModel) -> TileMemoryEstimate:
    c = model.config
    t, s = c.tile_size, c.baked_latent_size
    vae = model.vae
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
) -> tuple[torch.Tensor, RunStats]:
    """Super-resolve a ComfyUI IMAGE batch. ``tile_batch_size=0`` selects automatically."""
    import comfy.model_management as mm
    import comfy.utils

    validate_image(image)
    timestep = cfg.resolve_timestep(upscale_factor, creativity)
    index = model.config.timestep_index(timestep)
    if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 0:
        raise ValueError(f"tile_batch_size must be 0 (auto) or a positive integer, got {tile_batch_size!r}")
    if accumulation_device not in ACCUMULATION_MODES:
        raise ValueError(f"accumulation_device must be one of {ACCUMULATION_MODES}, got {accumulation_device!r}")

    batch, height, width, channels = image.shape
    alpha = image[..., 3:4] if channels == 4 else None
    rgb = image[..., :3]
    grid = make_grid(height, width, upscale_factor, model.config.tile_size, overlap)
    total_tiles = batch * grid.num_tiles

    estimate = estimate_tile_memory(model)
    per_tile = estimate.per_tile
    acc_bytes = grid.accumulation_bytes()
    requested = tile_batch_size if tile_batch_size > 0 else 1
    gpu_acc = 2 * acc_bytes if accumulation_device == "gpu" else 0
    mm.load_models_gpu([model.patcher, model.vae.patcher], memory_required=requested * per_tile + gpu_acc)

    device = model.patcher.load_device
    reserved = mm.extra_reserved_memory() if hasattr(mm, "extra_reserved_memory") else 0
    free = mm.get_free_memory(device) if device.type != "cpu" else 0
    if tile_batch_size > 0:
        batch_size = min(tile_batch_size, total_tiles)
    elif device.type == "cpu":
        batch_size = 1
    else:
        batch_size = auto_tile_batch(free, reserved + gpu_acc, per_tile, total_tiles)
    intermediate = mm.intermediate_device()
    acc_device = choose_accumulation_device(
        accumulation_device, device, intermediate, acc_bytes, free - reserved - batch_size * per_tile
    )
    logger.info(
        "Kandinsky ImageSR: %d image(s) %dx%d -> %dx%d, %d tiles (%dx%d grid), t=%s, tile batch %d "
        "(~%s per tile), accumulator %s on %s",
        batch,
        width,
        height,
        grid.out_width,
        grid.out_height,
        total_tiles,
        grid.cols,
        grid.rows,
        timestep,
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
            output, stats = TiledRestorer(_ComfyBackend(model, rope), grid).run(rgb, options)
    finally:
        del rope

    if alpha is not None:
        # Alpha is not part of the generative model: resize it bilinearly to the
        # exact output size and append it unchanged in meaning.
        resized = F.interpolate(
            alpha.movedim(-1, 1).float(), size=(grid.out_height, grid.out_width), mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)
        output = torch.cat([output, resized.movedim(1, -1).to(output)], dim=-1)

    out_dtype = mm.intermediate_dtype() if hasattr(mm, "intermediate_dtype") else torch.float32
    if output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output, stats

"""Configuration, constants and validation for Kandinsky ImageSR 2.0 bundles.

Every value here mirrors the published Hugging Face bundle
(``kandinskylab/Kandinsky-ImageSR-2.0``) and the official inference code
(``kandinskylab/kandinsky-image-sr``). Validation is deliberately strict: a
bundle that this implementation cannot execute faithfully is rejected with an
explicit reason instead of being loaded approximately.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSFORMER_CLASS_NAME = "KandinskyImageSRBakedTransformer3DModel"
PIPELINE_CLASS_NAME = "KandinskyImageSRPipeline"
VAE_CLASS_NAME = "AutoencoderKL"

#: Calibrated creativity presets of the official pipeline, indexed by level 0..8.
CREATIVITY_TIMESTEPS: dict[int, tuple[float, ...]] = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}
#: Creativity level that reproduces the official default timestep of each factor.
DEFAULT_CREATIVITY = 4
SUPPORTED_UPSCALE_FACTORS: tuple[int, ...] = (2, 4)
MAX_CREATIVITY = 8

#: Tolerance used by the official implementation to match a timestep to a baked entry.
TIMESTEP_MATCH_TOLERANCE = 1e-6

#: The VAE is the FLUX.1 autoencoder: 8x spatial compression, 16 latent channels.
VAE_SPATIAL_COMPRESSION = 8
LATENT_CHANNELS = 16

#: Default fraction of each low-resolution tile that overlaps its neighbour.
DEFAULT_OVERLAP = 0.25
#: Supported overlap range. The reference accepts [0, 1), but with zero overlap
#: interior tile borders receive zero Hann weight and turn black, and very large
#: overlaps only multiply the tile count. The range below keeps every output
#: pixel covered by a strictly positive blend weight.
MIN_OVERLAP = 0.05
MAX_OVERLAP = 0.75


class BundleError(ValueError):
    """Raised when a model bundle is missing, incomplete or unsupported."""


@dataclass(frozen=True)
class TransformerConfig:
    """Validated architecture of the baked ImageSR transformer."""

    in_visual_dim: int
    out_visual_dim: int
    time_dim: int
    patch_size: tuple[int, int, int]
    model_dim: int
    ff_dim: int
    num_visual_blocks: int
    axes_dims: tuple[int, int, int]
    baked_timesteps: tuple[float, ...]
    baked_latent_size: int

    @property
    def head_dim(self) -> int:
        return sum(self.axes_dims)

    @property
    def num_heads(self) -> int:
        return self.model_dim // self.head_dim

    @property
    def tile_size(self) -> int:
        """Output tile side in pixels (latent side times the VAE compression)."""
        return self.baked_latent_size * VAE_SPATIAL_COMPRESSION

    @property
    def token_grid(self) -> tuple[int, int, int]:
        """Token grid (duration, height, width) after patch embedding."""
        pt, ph, pw = self.patch_size
        return (1 // pt, self.baked_latent_size // ph, self.baked_latent_size // pw)

    @property
    def num_tokens(self) -> int:
        return math.prod(self.token_grid)

    @property
    def patch_dim(self) -> int:
        return math.prod(self.patch_size) * self.in_visual_dim

    def timestep_index(self, timestep: float) -> int:
        """Return the baked-table row for ``timestep``.

        Matching uses the official absolute tolerance on Python floats taken
        from the bundle configuration, never on values that went through a
        reduced-precision tensor, so the lookup is exact and dtype independent.
        """
        best = min(range(len(self.baked_timesteps)), key=lambda i: abs(self.baked_timesteps[i] - timestep))
        if abs(self.baked_timesteps[best] - timestep) > TIMESTEP_MATCH_TOLERANCE:
            raise BundleError(
                f"timestep {timestep} is not baked into this model; available values: {list(self.baked_timesteps)}"
            )
        return best

    def expected_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Exact tensor names and shapes of a complete transformer checkpoint."""
        d, t, f = self.model_dim, self.time_dim, self.ff_dim
        n = len(self.baked_timesteps)
        shapes: dict[str, tuple[int, ...]] = {
            "time_embed_table": (n, t),
            "visual_embeddings.in_layer.weight": (d, self.patch_dim),
            "visual_embeddings.in_layer.bias": (d,),
            "out_layer.modulation.out_layer.weight": (2 * d, t),
            "out_layer.modulation.out_layer.bias": (2 * d,),
            "out_layer.out_layer.weight": (math.prod(self.patch_size) * self.out_visual_dim, d),
            "out_layer.out_layer.bias": (math.prod(self.patch_size) * self.out_visual_dim,),
        }
        for i in range(self.num_visual_blocks):
            p = f"visual_transformer_blocks.{i}."
            shapes[p + "visual_modulation.out_layer.weight"] = (9 * d, t)
            shapes[p + "visual_modulation.out_layer.bias"] = (9 * d,)
            for name in ("to_query", "to_key", "to_value", "out_layer"):
                shapes[p + f"self_attention.{name}.weight"] = (d, d)
                shapes[p + f"self_attention.{name}.bias"] = (d,)
            shapes[p + "self_attention.query_norm.weight"] = (self.head_dim,)
            shapes[p + "self_attention.key_norm.weight"] = (self.head_dim,)
            shapes[p + "feed_forward.in_layer.weight"] = (f, d)
            shapes[p + "feed_forward.out_layer.weight"] = (d, f)
            shapes[p + "xattn_bias_table"] = (n, d)
        return shapes

    @staticmethod
    def is_fp32_tensor(name: str) -> bool:
        """Tensors the reference keeps in FP32 (``_keep_in_fp32_modules``)."""
        return name.endswith(("time_embed_table", "xattn_bias_table")) or ".modulation." in name or (
            "visual_modulation." in name
        )


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise BundleError(message)


def _int_tuple(value: Any, length: int, field: str) -> tuple[int, ...]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value),
        f"transformer config field {field!r} must be a list of {length} integers, got {value!r}",
    )
    return tuple(int(v) for v in value)


def _positive_int(cfg: dict[str, Any], field: str) -> int:
    value = cfg.get(field)
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"transformer config field {field!r} must be a positive integer, got {value!r}",
    )
    return int(value)


def parse_transformer_config(cfg: dict[str, Any]) -> TransformerConfig:
    """Validate a ``transformer/config.json`` dictionary."""
    _require(isinstance(cfg, dict), "transformer config must be a JSON object")
    class_name = cfg.get("_class_name")
    _require(
        class_name == TRANSFORMER_CLASS_NAME,
        f"unsupported transformer class {class_name!r}; expected {TRANSFORMER_CLASS_NAME!r}. "
        "Standard Kandinsky 5 transformers are not interchangeable with the baked ImageSR model.",
    )
    _require(cfg.get("attention_type", "regular") == "regular", "only regular (dense) attention is supported")
    _require(cfg.get("visual_cond", False) is False, "visual_cond=True bundles are not supported")

    patch = _int_tuple(cfg.get("patch_size"), 3, "patch_size")
    _require(patch == (1, 2, 2), f"patch_size must be [1, 2, 2], got {list(patch)}")
    axes = _int_tuple(cfg.get("axes_dims"), 3, "axes_dims")
    _require(all(a > 0 and a % 2 == 0 for a in axes), f"axes_dims must be positive and even, got {list(axes)}")

    in_dim = _positive_int(cfg, "in_visual_dim")
    out_dim = _positive_int(cfg, "out_visual_dim")
    _require(
        in_dim == LATENT_CHANNELS and out_dim == LATENT_CHANNELS,
        f"in/out visual dims must equal the FLUX VAE latent channels ({LATENT_CHANNELS}), got {in_dim}/{out_dim}",
    )
    model_dim = _positive_int(cfg, "model_dim")
    head_dim = sum(axes)
    _require(model_dim % head_dim == 0, f"model_dim {model_dim} is not divisible by head_dim {head_dim}")
    latent = _positive_int(cfg, "baked_latent_size")
    _require(latent % 2 == 0, f"baked_latent_size must be even, got {latent}")
    _require(
        latent // 2 <= 128,
        "baked_latent_size exceeds the 128-position RoPE table of the Kandinsky 5 architecture",
    )

    raw_steps = cfg.get("baked_timesteps")
    _require(
        isinstance(raw_steps, (list, tuple)) and len(raw_steps) > 0,
        "transformer config must list baked_timesteps",
    )
    _require(
        all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_steps),
        "baked_timesteps must be numbers",
    )
    steps = tuple(float(v) for v in raw_steps)
    _require(all(math.isfinite(v) and 0.0 < v < 1.0 for v in steps), "baked_timesteps must lie in (0, 1)")
    _require(
        all(b - a > 2 * TIMESTEP_MATCH_TOLERANCE for a, b in zip(steps, steps[1:])),
        "baked_timesteps must be strictly increasing and well separated",
    )

    config = TransformerConfig(
        in_visual_dim=in_dim,
        out_visual_dim=out_dim,
        time_dim=_positive_int(cfg, "time_dim"),
        patch_size=patch,  # type: ignore[arg-type]
        model_dim=model_dim,
        ff_dim=_positive_int(cfg, "ff_dim"),
        num_visual_blocks=_positive_int(cfg, "num_visual_blocks"),
        axes_dims=axes,  # type: ignore[arg-type]
        baked_timesteps=steps,
        baked_latent_size=latent,
    )
    for factor, values in CREATIVITY_TIMESTEPS.items():
        for level, t in enumerate(values):
            try:
                config.timestep_index(t)
            except BundleError as error:
                raise BundleError(
                    f"bundle lacks the baked entry for x{factor} creativity {level} (t={t})"
                ) from error
    return config


def validate_vae_config(cfg: dict[str, Any]) -> float:
    """Validate ``vae/config.json`` and return its latent scaling factor."""
    _require(isinstance(cfg, dict), "VAE config must be a JSON object")
    _require(cfg.get("_class_name") == VAE_CLASS_NAME, f"VAE class must be {VAE_CLASS_NAME!r}")
    _require(cfg.get("latent_channels") == LATENT_CHANNELS, f"VAE must have {LATENT_CHANNELS} latent channels")
    _require(cfg.get("in_channels", 3) == 3 and cfg.get("out_channels", 3) == 3, "VAE must be RGB")
    blocks = cfg.get("block_out_channels")
    _require(
        isinstance(blocks, list) and len(blocks) == 4,
        "VAE must have four resolution levels (8x spatial compression)",
    )
    _require(not cfg.get("use_quant_conv", False) and not cfg.get("use_post_quant_conv", False),
             "VAE with quant convolutions is not the FLUX autoencoder")
    scale = cfg.get("scaling_factor")
    _require(
        isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0,
        f"invalid VAE scaling_factor {scale!r}",
    )
    return float(scale)


def validate_pipeline_config(cfg: dict[str, Any], transformer: TransformerConfig) -> None:
    """Check ``model_index.json`` against the geometry this implementation executes."""
    _require(isinstance(cfg, dict), "model_index.json must be a JSON object")
    _require(
        cfg.get("_class_name") == PIPELINE_CLASS_NAME,
        f"model_index.json describes {cfg.get('_class_name')!r}, not {PIPELINE_CLASS_NAME!r}",
    )
    factors = cfg.get("upscale_factors", list(SUPPORTED_UPSCALE_FACTORS))
    _require(
        isinstance(factors, list) and sorted(factors) == sorted(SUPPORTED_UPSCALE_FACTORS),
        f"unsupported upscale_factors {factors!r}",
    )
    tile = cfg.get("tile_size", transformer.tile_size)
    _require(tile == transformer.tile_size, f"tile_size {tile} does not match the baked latent size")
    scale = cfg.get("scale_factor", [1.0, 1.0, 1.0])
    _require(
        isinstance(scale, list) and [float(v) for v in scale] == [1.0, 1.0, 1.0],
        f"unsupported RoPE scale_factor {scale!r}",
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as error:
        raise BundleError(f"missing required file: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise BundleError(f"{path} must contain a JSON object")
    return data


def resolve_timestep(upscale_factor: int, creativity: int) -> float:
    """Map a factor and creativity level to the calibrated official timestep."""
    if upscale_factor not in CREATIVITY_TIMESTEPS:
        raise ValueError(f"upscale_factor must be one of {SUPPORTED_UPSCALE_FACTORS}, got {upscale_factor!r}")
    if isinstance(creativity, bool) or not isinstance(creativity, int) or not 0 <= creativity <= MAX_CREATIVITY:
        raise ValueError(f"creativity must be an integer from 0 to {MAX_CREATIVITY}, got {creativity!r}")
    return CREATIVITY_TIMESTEPS[upscale_factor][creativity]

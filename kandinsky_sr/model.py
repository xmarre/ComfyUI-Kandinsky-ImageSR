"""Native Kandinsky ImageSR 2.0 baked transformer.

This is a numerically faithful port of ``KandinskyImageSRBakedTransformer3DModel``
from ``kandinskylab/kandinsky-image-sr`` (MIT) and the Kandinsky 5 building blocks
it uses from Diffusers (``transformer_kandinsky.py``, Apache-2.0). Parameter names
match the published checkpoint so the original shards load without remapping.

Precision contract (BF16 execution) mirrors the official CUDA path, which runs the
transformer under ``torch.autocast(dtype=bfloat16)``:

* every linear projection, including the modulation layers whose weights are
  stored in FP32, computes in BF16 (autocast casts weight and input to BF16);
* LayerNorm and the query/key RMSNorm compute in FP32 (autocast FP32 ops);
* scale/shift modulation, gated residual updates and the baked cross-attention
  bias are computed in FP32 and rounded back to the activation dtype;
* RoPE is applied in FP32 and rounded to BF16;
* the baked time embedding and cross-attention bias tables stay FP32.

ComfyUI's native Kandinsky 5 blocks are not reused: they apply modulation and
residual updates in the activation dtype and normalise queries/keys in BF16 with
a BF16 epsilon, which is not equivalent to the reference computation.

Linear and RMSNorm layers use ComfyUI operation classes when ComfyUI is present,
so ModelPatcher low-VRAM partial loading and weight casting work unchanged.
Attention goes through ComfyUI's optimized attention dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


# --------------------------------------------------------------------------- operations


class _CastLinear(nn.Linear):
    """Linear layer that casts its weight to the input dtype (autocast semantics)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=input.dtype)
        return F.linear(input, self.weight.to(device=input.device, dtype=input.dtype), bias)


class _CastRMSNorm(nn.RMSNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=input.device, dtype=input.dtype)
        return F.rms_norm(input, self.normalized_shape, weight, self.eps)


def torch_operations() -> SimpleNamespace:
    """Plain PyTorch operations, used when ComfyUI is unavailable (tests, tools)."""
    return SimpleNamespace(Linear=nn.Linear, CastLinear=_CastLinear, CastRMSNorm=_CastRMSNorm)


def comfy_operations() -> SimpleNamespace:
    """ComfyUI operation classes.

    Large projections use ``disable_weight_init`` (no casting while fully loaded,
    automatic casting when ModelPatcher loads the model partially). Modulation and
    RMSNorm layers always cast to the input dtype because their FP32 (modulation)
    or BF16 (norm) weights deliberately differ from the dtype they compute in.
    """
    import comfy.ops

    return SimpleNamespace(
        Linear=comfy.ops.disable_weight_init.Linear,
        CastLinear=comfy.ops.manual_cast.Linear,
        CastRMSNorm=comfy.ops.manual_cast.RMSNorm,
    )


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Reference attention: ``q, k, v`` are ``(B, H, N, D)``; returns ``(B, N, H*D)``."""
    out = F.scaled_dot_product_attention(q, k, v)
    b, h, n, d = out.shape
    return out.transpose(1, 2).reshape(b, n, h * d)


def comfy_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """ComfyUI optimized attention dispatch (respects the configured backend)."""
    from comfy.ldm.modules.attention import optimized_attention

    return optimized_attention(q, k, v, heads, skip_reshape=True, transformer_options={})


# --------------------------------------------------------------------------- functional pieces


def get_freqs(dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Kandinsky 5 frequency ladder, computed in FP32 exactly like the reference."""
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def build_rope(config: TransformerConfig, device: torch.device) -> torch.Tensor:
    """3D rotary table for the fixed token grid, shape ``(1, N, 1, D/2, 2, 2)`` FP32.

    Positions are the integer token coordinates with a unit scale factor, as in the
    official pipeline (``visual_rope_pos = arange`` and ``scale_factor = (1, 1, 1)``).
    The FP32 argument table is computed on CPU so it is identical on every device.
    """
    duration, height, width = config.token_grid
    parts = []
    for axis, (axes_dim, length) in enumerate(zip(config.axes_dims, (duration, height, width))):
        freq = get_freqs(axes_dim // 2)
        args = torch.outer(torch.arange(length, dtype=torch.float32), freq)
        view = [1, 1, 1, -1]
        view[axis] = length
        parts.append(args.view(*view).expand(duration, height, width, -1))
    args = torch.cat(parts, dim=-1)
    cosine, sine = torch.cos(args), torch.sin(args)
    rope = torch.stack([cosine, -sine, sine, cosine], dim=-1).view(*args.shape, 2, 2)
    return rope.reshape(1, duration * height * width, 1, -1, 2, 2).to(device)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs in FP32; ``x`` is ``(B, N, H, D)``.

    Matches the reference ``(rope * x_).sum(-1)`` evaluation order. In BF16 mode the
    result is rounded to BF16 as in the reference; higher-precision modes keep their
    own dtype.
    """
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    return (rope * x_).sum(dim=-1).reshape(x.shape).to(x.dtype)


def modulated_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``LayerNorm(x) * (1 + scale) + shift`` in FP32, rounded to ``x.dtype``."""
    normed = F.layer_norm(x.float(), (x.shape[-1],))
    return (normed * (scale.float() + 1.0) + shift.float()).to(x.dtype)


def gated_residual(x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() + gate.float() * update.float()).to(x.dtype)


# --------------------------------------------------------------------------- modules


class Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int, ops: SimpleNamespace, **factory: Any) -> None:
        super().__init__()
        self.out_layer = ops.CastLinear(time_dim, num_params * model_dim, **factory)

    def forward(self, time_embed: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        # SiLU runs on the FP32 embedding; the projection computes in the activation
        # dtype with the FP32 weights cast to it, exactly as under autocast.
        return self.out_layer(F.silu(time_embed).to(compute_dtype))


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, head_dim: int, ops: SimpleNamespace, **factory: Any) -> None:
        super().__init__()
        self.num_heads = model_dim // head_dim
        self.to_query = ops.Linear(model_dim, model_dim, bias=True, **factory)
        self.to_key = ops.Linear(model_dim, model_dim, bias=True, **factory)
        self.to_value = ops.Linear(model_dim, model_dim, bias=True, **factory)
        self.query_norm = ops.CastRMSNorm(head_dim, **factory)
        self.key_norm = ops.CastRMSNorm(head_dim, **factory)
        self.out_layer = ops.Linear(model_dim, model_dim, bias=True, **factory)

    def _qk(self, x: torch.Tensor, proj: nn.Module, norm: nn.Module, rope: torch.Tensor) -> torch.Tensor:
        y = proj(x)
        y = y.view(*y.shape[:-1], self.num_heads, -1)
        # FP32 RMSNorm with the FP32 machine epsilon (nn.RMSNorm(eps=None) on FP32 input).
        y = norm(y.float()).to(y.dtype)
        return apply_rope(y, rope)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attention: AttentionFn) -> torch.Tensor:
        q = self._qk(x, self.to_query, self.query_norm, rope).transpose(1, 2)
        k = self._qk(x, self.to_key, self.key_norm, rope).transpose(1, 2)
        v = self.to_value(x)
        v = v.view(*v.shape[:-1], self.num_heads, -1).transpose(1, 2)
        out = attention(q, k, v, self.num_heads)
        del q, k, v
        return self.out_layer(out)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ops: SimpleNamespace, **factory: Any) -> None:
        super().__init__()
        self.in_layer = ops.Linear(model_dim, ff_dim, bias=False, **factory)
        self.out_layer = ops.Linear(ff_dim, model_dim, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(x)))


class BakedBlock(nn.Module):
    """Kandinsky 5 visual block whose zero-text cross-attention is a baked bias table."""

    def __init__(self, config: TransformerConfig, ops: SimpleNamespace, fp32_factory: dict, factory: dict) -> None:
        super().__init__()
        d = config.model_dim
        self.visual_modulation = Modulation(config.time_dim, d, 9, ops, **fp32_factory)
        self.self_attention = SelfAttention(d, config.head_dim, ops, **factory)
        self.feed_forward = FeedForward(d, config.ff_dim, ops, **factory)
        self.register_buffer(
            "xattn_bias_table",
            torch.empty(len(config.baked_timesteps), d, **fp32_factory),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        timestep_indices: torch.Tensor,
        rope: torch.Tensor,
        attention: AttentionFn,
    ) -> torch.Tensor:
        params = self.visual_modulation(time_embed, x.dtype).unsqueeze(1)
        self_attn_params, _, ff_params = torch.chunk(params, 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.self_attention(modulated_norm(x, shift, scale), rope, attention)
        x = gated_residual(x, out, gate)
        del out

        table = self.xattn_bias_table
        bias = table.index_select(0, timestep_indices.to(table.device)).to(x.device).unsqueeze(1)
        x = (x.float() + bias).to(x.dtype)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.feed_forward(modulated_norm(x, shift, scale))
        return gated_residual(x, out, gate)


class OutLayer(nn.Module):
    def __init__(self, config: TransformerConfig, ops: SimpleNamespace, fp32_factory: dict, factory: dict) -> None:
        super().__init__()
        self.modulation = Modulation(config.time_dim, config.model_dim, 2, ops, **fp32_factory)
        self.out_layer = ops.Linear(
            config.model_dim, math.prod(config.patch_size) * config.out_visual_dim, bias=True, **factory
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed, x.dtype).unsqueeze(1), 2, dim=-1)
        return self.out_layer(modulated_norm(x, shift, scale))


class KandinskyImageSRTransformer(nn.Module):
    """Baked ImageSR DiT operating on fixed ``baked_latent_size`` latent tiles.

    ``forward`` takes channel-first latents ``(B, 16, S, S)`` already multiplied by
    the VAE scaling factor and returns the flow velocity in the same layout as FP32.
    """

    def __init__(
        self,
        config: TransformerConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        operations: SimpleNamespace | None = None,
        attention: AttentionFn | None = None,
    ) -> None:
        super().__init__()
        ops = operations or torch_operations()
        factory = {"device": device, "dtype": dtype}
        fp32_factory = {"device": device, "dtype": torch.float32}
        self.config = config
        self.compute_dtype = dtype
        self.attention: AttentionFn = attention or sdpa_attention
        self.visual_embeddings = _VisualEmbeddings(config, ops, factory)
        self.visual_transformer_blocks = nn.ModuleList(
            [BakedBlock(config, ops, fp32_factory, factory) for _ in range(config.num_visual_blocks)]
        )
        self.out_layer = OutLayer(config, ops, fp32_factory, factory)
        self.register_buffer("time_embed_table", torch.empty(len(config.baked_timesteps), config.time_dim, **fp32_factory))

    def forward(self, latents: torch.Tensor, timestep_index: int, rope: torch.Tensor | None = None) -> torch.Tensor:
        """``rope`` may be precomputed with :func:`build_rope` and reused across calls."""
        cfg = self.config
        size = cfg.baked_latent_size
        if latents.ndim != 4 or latents.shape[1] != cfg.in_visual_dim or tuple(latents.shape[2:]) != (size, size):
            raise ValueError(
                f"baked transformer expects latents shaped (B, {cfg.in_visual_dim}, {size}, {size}), "
                f"got {tuple(latents.shape)}"
            )
        if not 0 <= timestep_index < len(cfg.baked_timesteps):
            raise ValueError(f"timestep index {timestep_index} outside the baked table")
        batch = latents.shape[0]
        device = latents.device
        dtype = self.compute_dtype

        indices = torch.full((batch,), int(timestep_index), dtype=torch.long, device=device)
        table = self.time_embed_table
        time_embed = table.index_select(0, indices.to(table.device)).to(device=device, dtype=torch.float32)

        # (B, C, H, W) -> (B, T=1, H, W, C), as the reference pipeline feeds the model.
        hidden = latents.permute(0, 2, 3, 1).unsqueeze(1).to(dtype)
        x = self.visual_embeddings(hidden)
        grid = x.shape[1:4]
        x = x.flatten(1, 3)
        if rope is None:
            rope = build_rope(cfg, device)
        rope = rope.to(device)

        for block in self.visual_transformer_blocks:
            x = block(x, time_embed, indices, rope, self.attention)

        x = self.out_layer(x, time_embed)
        pt, ph, pw = cfg.patch_size
        out = x.view(batch, *grid, cfg.out_visual_dim, pt, ph, pw)
        # (B, T, H, W, C, pt, ph, pw) -> (B, C, T*pt, H*ph, W*pw); T*pt == 1.
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(batch, cfg.out_visual_dim, size, size)
        return out.float()


class _VisualEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig, ops: SimpleNamespace, factory: dict) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_layer = ops.Linear(config.patch_dim, config.model_dim, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        pt, ph, pw = self.patch_size
        x = x.view(b, t // pt, pt, h // ph, ph, w // pw, pw, c).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)
        return self.in_layer(x)

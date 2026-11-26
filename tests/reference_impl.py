"""Independent oracle: upstream reference code, copied with minimal changes.

Sources (pinned):

* ``kandinskylab/kandinsky-image-sr`` @ 6c2405796cf7576b0cffff27acd103d667b91ca8
  (``baked_transformer.py``, ``tiling.py``, ``pipeline.py``), MIT License,
  Copyright (c) 2026 kandinskylab.
* ``huggingface/diffusers`` 0.43.0 ``models/transformers/transformer_kandinsky.py``,
  Apache License 2.0, Copyright 2025 The Kandinsky Team and The HuggingFace Team.

Changes: Diffusers base classes (ModelMixin, ConfigMixin, attention dispatch) are
replaced by plain ``nn.Module`` and ``F.scaled_dot_product_attention`` (the
"native" Diffusers backend), sparse attention paths are removed, and the
pipeline's tile loop is reduced to a function taking an ``infer_tiles`` callable.
The arithmetic of every retained line is unchanged. This module is used only by
tests to check the native implementation against the upstream computation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------- diffusers (Apache-2.0)


def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)
    return freqs


def fractal_flatten(x, rope, shape, block_mask=False):
    x = x.flatten(1, 3)
    rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(x, shape, block_mask=False):
    x = x.reshape(*shape, *x.shape[2:])
    return x


class Kandinsky5VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x) -> torch.Tensor:
        batch_size, duration, height, width, dim = x.shape
        x = (
            x.view(
                batch_size,
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class Kandinsky5RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)) -> torch.Tensor:
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).repeat(batch_size, 1, height, width, 1),
                args_h.view(1, 1, height, 1, -1).repeat(batch_size, duration, 1, width, 1),
                args_w.view(1, 1, 1, width, -1).repeat(batch_size, duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky5Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def forward(self, x) -> torch.Tensor:
        return self.out_layer(self.activation(x))


class Kandinsky5Attention(nn.Module):
    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, hidden_states, rotary_emb=None, sparse_params=None):
        attn = self
        query = attn.to_query(hidden_states)
        key = attn.to_key(hidden_states)
        value = attn.to_value(hidden_states)

        shape = query.shape[:-1]
        query = query.reshape(*shape, attn.num_heads, -1)
        key = key.reshape(*shape, attn.num_heads, -1)
        value = value.reshape(*shape, attn.num_heads, -1)

        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)

        def apply_rotary(x, rope):
            x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
            x_out = (rope * x_).sum(dim=-1)
            return x_out.reshape(*x.shape).to(torch.bfloat16)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).type_as(query)
            key = apply_rotary(key, rotary_emb).type_as(key)

        # Diffusers "native" attention backend.
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        hidden_states = hidden_states.flatten(-2, -1)

        attn_out = attn.out_layer(hidden_states)
        return attn_out


class Kandinsky5FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class Kandinsky5OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky5Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed, text_embed, time_embed) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)

        visual_embed = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0) + shift.float()[:, None, None]
        ).type_as(visual_embed)

        x = self.out_layer(visual_embed)

        batch_size, duration, height, width, _ = x.shape
        x = (
            x.view(
                batch_size,
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        return x


# ----------------------------------------------------------------- kandinsky-image-sr (MIT)


class KandinskyImageSRBakedBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, num_timesteps):
        super().__init__()
        self.visual_modulation = Kandinsky5Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5Attention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)
        self.register_buffer("xattn_bias_table", torch.zeros(num_timesteps, model_dim, dtype=torch.float32))

    def forward(self, visual_embed, time_embed, rope, sparse_params, timestep_indices):
        self_attn_params, _, ff_params = torch.chunk(self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = (self.self_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            visual_embed
        )
        visual_out = self.self_attention(visual_out, rotary_emb=rope, sparse_params=sparse_params)
        visual_embed = (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)

        baked_bias = self.xattn_bias_table.index_select(0, timestep_indices).unsqueeze(1)
        visual_embed = (visual_embed.float() + baked_bias).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = (self.feed_forward_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            visual_embed
        )
        visual_out = self.feed_forward(visual_out)
        return (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)


class KandinskyImageSRBakedTransformer3DModel(nn.Module):
    _keep_in_fp32_modules = ["modulation", "visual_modulation", "time_embed_table", "xattn_bias_table"]

    def __init__(
        self,
        in_visual_dim=16,
        out_visual_dim=16,
        time_dim=512,
        patch_size=(1, 2, 2),
        model_dim=2560,
        ff_dim=10240,
        num_visual_blocks=50,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_type="regular",
        baked_timesteps=(0.15, 0.244),
        baked_latent_size=128,
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim
        self.baked_timesteps = tuple(float(value) for value in baked_timesteps)
        self.baked_latent_size = int(baked_latent_size)

        self.visual_embeddings = Kandinsky5VisualEmbeddings(visual_embed_dim, model_dim, patch_size)
        self.visual_rope_embeddings = Kandinsky5RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                KandinskyImageSRBakedBlock(model_dim, time_dim, ff_dim, head_dim, len(self.baked_timesteps))
                for _ in range(num_visual_blocks)
            ]
        )
        self.out_layer = Kandinsky5OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        self.register_buffer("time_embed_table", torch.zeros(len(self.baked_timesteps), time_dim, dtype=torch.float32))

    def _timestep_indices(self, timestep):
        requested = timestep.float() / 1000.0
        available = torch.tensor(self.baked_timesteps, device=timestep.device, dtype=torch.float32)
        distance, indices = (requested[:, None] - available[None]).abs().min(dim=1)
        if bool((distance > 1e-6).any()):
            raise ValueError("timestep not baked")
        return indices

    def forward(self, hidden_states, timestep, visual_rope_pos, scale_factor=(1.0, 1.0, 1.0)):
        timestep_indices = self._timestep_indices(timestep)
        time_embed = self.time_embed_table.index_select(0, timestep_indices)
        visual_embed = self.visual_embeddings(hidden_states)
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape, block_mask=False)

        for block in self.visual_transformer_blocks:
            visual_embed = block(visual_embed, time_embed, visual_rope, None, timestep_indices)

        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=False)
        return self.out_layer(visual_embed, None, time_embed)


def reference_velocity(model, lq, timestep):
    """Pipeline-side transformer call of ``KandinskyImageSRPipeline._infer_tiles``."""
    batch, _, height, width = lq.shape
    latent = lq.permute(0, 2, 3, 1).unsqueeze(1)
    visual_rope_pos = (torch.arange(1), torch.arange(height // 2), torch.arange(width // 2))
    model_t = torch.full((batch,), timestep * 1000.0, dtype=torch.float32)
    prediction = model(latent.to(dtype=torch.bfloat16), model_t, visual_rope_pos)
    return prediction.squeeze(1).permute(0, 3, 1, 2).float()


# ----------------------------------------------------------------- tiling (MIT)


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def pad_to_min_tile(image, tile):
    _, _, height, width = image.shape
    pad_h = max(0, tile - height)
    pad_w = max(0, tile - width)
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate"), height, width


def boundary_aware_window(height, width, *, top, bottom, left, right, device):
    wy = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(height, device=device) / height)
    wx = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(width, device=device) / width)
    if top:
        wy[: height // 2] = 1.0
    if bottom:
        wy[height // 2 :] = 1.0
    if left:
        wx[: width // 2] = 1.0
    if right:
        wx[width // 2 :] = 1.0
    return wy[:, None] * wx[None, :]


def reference_tiled_call(input_tensor, infer_tiles, *, upscale_factor, tile_size=1024, overlap=0.25, tile_batch_size=1):
    """Tile loop of ``KandinskyImageSRPipeline.__call__`` for one ``(1, 3, H, W)`` image.

    ``infer_tiles(pixels)`` receives the clamped bicubic tiles and returns decoded
    tiles in [0, 1], standing in for ``_infer_tiles``.
    """
    device = torch.device("cpu")
    lq_tile = tile_size // upscale_factor
    stride = max(1, int(lq_tile * (1.0 - overlap)))
    padded, original_h, original_w = pad_to_min_tile(input_tensor, lq_tile)
    tops = tile_positions(padded.shape[-2], lq_tile, stride)
    lefts = tile_positions(padded.shape[-1], lq_tile, stride)
    cells = [(row, col) for row in range(len(tops)) for col in range(len(lefts))]
    output_h, output_w = padded.shape[-2] * upscale_factor, padded.shape[-1] * upscale_factor
    accumulator = torch.zeros(1, 3, output_h, output_w, device=device)
    weights = torch.zeros(1, 1, output_h, output_w, device=device)

    for start in range(0, len(cells), tile_batch_size):
        chunk = cells[start : start + tile_batch_size]
        tiles = []
        for row, col in chunk:
            top, left = tops[row], lefts[col]
            tile = padded[:, :, top : top + lq_tile, left : left + lq_tile]
            tiles.append(F.interpolate(tile, size=(tile_size, tile_size), mode="bicubic", align_corners=False))
        predictions = infer_tiles(torch.cat(tiles).clamp(0.0, 1.0))
        for index, (row, col) in enumerate(chunk):
            window = boundary_aware_window(
                tile_size,
                tile_size,
                top=row == 0,
                bottom=row == len(tops) - 1,
                left=col == 0,
                right=col == len(lefts) - 1,
                device=device,
            )[None, None]
            top_hr, left_hr = tops[row] * upscale_factor, lefts[col] * upscale_factor
            accumulator[:, :, top_hr : top_hr + tile_size, left_hr : left_hr + tile_size] += (
                predictions[index : index + 1] * window
            )
            weights[:, :, top_hr : top_hr + tile_size, left_hr : left_hr + tile_size] += window

    return (accumulator / weights.clamp_min(1e-6))[:, :, : original_h * upscale_factor, : original_w * upscale_factor]

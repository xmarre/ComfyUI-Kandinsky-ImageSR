"""Native transformer versus the upstream reference computation."""

from __future__ import annotations

import pytest
import torch

from kandinsky_sr.config import CREATIVITY_TIMESTEPS, parse_transformer_config
from kandinsky_sr.model import KandinskyImageSRTransformer, apply_rope, build_rope
from tests.helpers import random_state_dict, reference_model, small_transformer_config
from tests.reference_impl import Kandinsky5RoPE3D, reference_velocity


def _pair(seed=0, **overrides):
    cfg_dict = small_transformer_config(**overrides)
    cfg = parse_transformer_config(cfg_dict)
    state = random_state_dict(cfg, seed)
    ref = reference_model(cfg_dict, state)
    native = KandinskyImageSRTransformer(cfg, dtype=torch.bfloat16)
    native.load_state_dict(state, strict=True)
    return cfg, ref.eval(), native.eval()


def _latents(cfg, batch=2, seed=1):
    gen = torch.Generator().manual_seed(seed)
    size = cfg.baked_latent_size
    return torch.randn(batch, cfg.in_visual_dim, size, size, generator=gen)


@torch.no_grad()
@pytest.mark.parametrize("factor,level", [(2, 0), (2, 4), (2, 8), (4, 0), (4, 4), (4, 8)])
def test_bitwise_match_with_reference_under_autocast(factor, level):
    """Official CUDA inference runs under bf16 autocast; CPU autocast has the same op policy here."""
    cfg, ref, native = _pair()
    lq = _latents(cfg)
    t = CREATIVITY_TIMESTEPS[factor][level]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference_velocity(ref, lq, t)
    actual = native(lq, cfg.timestep_index(t))
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected), (actual - expected).abs().max()


@torch.no_grad()
def test_every_baked_timestep_selects_its_own_tables():
    cfg, ref, native = _pair()
    lq = _latents(cfg, batch=1)
    outputs = []
    for t in cfg.baked_timesteps:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = reference_velocity(ref, lq, t)
        actual = native(lq, cfg.timestep_index(t))
        assert torch.equal(actual, expected)
        outputs.append(actual)
    # distinct tables must produce distinct predictions
    for a, b in zip(outputs, outputs[1:]):
        assert not torch.equal(a, b)


@torch.no_grad()
def test_fp32_conditioning_tensors_are_preserved():
    cfg, _, native = _pair()
    for name, tensor in list(native.named_parameters()) + list(native.named_buffers()):
        expected = torch.float32 if cfg.is_fp32_tensor(name) else torch.bfloat16
        assert tensor.dtype == expected, name
    assert native.time_embed_table.dtype == torch.float32
    assert native.visual_transformer_blocks[0].xattn_bias_table.dtype == torch.float32


@torch.no_grad()
def test_baked_bias_is_added_after_attention_and_before_feed_forward():
    """Zeroing the attention and FFN gates leaves exactly residual + bias (rounded once)."""
    cfg, _, native = _pair(blocks=1)
    block = native.visual_transformer_blocks[0]
    d = cfg.model_dim
    with torch.no_grad():
        # gates are chunks 2 (attention) and 8 (ffn) of the 9 modulation parameters
        block.visual_modulation.out_layer.weight[2 * d : 3 * d].zero_()
        block.visual_modulation.out_layer.bias[2 * d : 3 * d].zero_()
        block.visual_modulation.out_layer.weight[8 * d : 9 * d].zero_()
        block.visual_modulation.out_layer.bias[8 * d : 9 * d].zero_()
    x = torch.randn(2, cfg.num_tokens, d).to(torch.bfloat16)
    idx = torch.tensor([3, 3])
    time_embed = native.time_embed_table.index_select(0, idx)
    rope = build_rope(cfg, x.device)
    out = block(x, time_embed, idx, rope, native.attention)
    expected = (x.float() + block.xattn_bias_table[3][None, None]).to(torch.bfloat16)
    assert torch.equal(out, expected)


@torch.no_grad()
def test_batch_elements_are_independent():
    cfg, _, native = _pair()
    lq = _latents(cfg, batch=3)
    idx = cfg.timestep_index(0.244)
    batched = native(lq, idx)
    for i in range(3):
        single = native(lq[i : i + 1], idx)
        torch.testing.assert_close(batched[i : i + 1], single, rtol=0, atol=2e-2)
        assert (batched[i : i + 1] - single).abs().mean() < 1e-3


def test_rope_matches_reference_coordinates():
    cfg = parse_transformer_config(small_transformer_config(latent=32))
    ref = Kandinsky5RoPE3D(cfg.axes_dims)
    _, h, w = cfg.token_grid
    expected = ref((1, 1, h, w), (torch.arange(1), torch.arange(h), torch.arange(w))).flatten(1, 3)
    actual = build_rope(cfg, torch.device("cpu"))
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_rope_rotates_adjacent_pairs_with_bf16_rounding():
    cfg = parse_transformer_config(small_transformer_config())
    rope = build_rope(cfg, torch.device("cpu"))
    x = torch.randn(1, cfg.num_tokens, cfg.num_heads, cfg.head_dim).to(torch.bfloat16)
    out = apply_rope(x, rope)
    assert out.dtype == torch.bfloat16
    xf = x.float().view(*x.shape[:-1], -1, 2)
    cos, sin = rope[..., 0, 0], rope[..., 1, 0]
    manual = torch.stack([cos * xf[..., 0] - sin * xf[..., 1], sin * xf[..., 0] + cos * xf[..., 1]], -1)
    assert torch.equal(out, manual.flatten(-2).to(torch.bfloat16))
    # token 0 has all-zero angles: identity
    assert torch.equal(out[:, 0], x[:, 0])


@torch.no_grad()
def test_rejects_invalid_inputs():
    cfg, _, native = _pair()
    s = cfg.baked_latent_size
    with pytest.raises(ValueError, match="expects latents"):
        native(torch.zeros(1, 16, s, s * 2), 0)
    with pytest.raises(ValueError, match="expects latents"):
        native(torch.zeros(1, 4, s, s), 0)
    with pytest.raises(ValueError, match="timestep index"):
        native(torch.zeros(1, 16, s, s), len(cfg.baked_timesteps))


@torch.no_grad()
def test_fp32_mode_is_close_to_bf16_reference():
    cfg_dict = small_transformer_config()
    cfg = parse_transformer_config(cfg_dict)
    state = random_state_dict(cfg)
    ref = reference_model(cfg_dict, state)
    native = KandinskyImageSRTransformer(cfg, dtype=torch.float32)
    native.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    lq = _latents(cfg)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference_velocity(ref, lq, 0.15)
    actual = native(lq, cfg.timestep_index(0.15))
    rel = (actual - expected).norm() / expected.norm()
    assert rel < 0.05

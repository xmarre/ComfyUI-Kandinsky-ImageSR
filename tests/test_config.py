"""Configuration parsing, creativity mapping and strict validation."""

from __future__ import annotations

import copy

import pytest

from kandinsky_sr.config import (
    CREATIVITY_TIMESTEPS,
    BundleError,
    parse_transformer_config,
    resolve_timestep,
    validate_pipeline_config,
    validate_vae_config,
)
from tests.helpers import published_model_index, published_transformer_config, published_vae_config

# Row of each creativity preset in the published baked table, read from
# transformer/config.json of the Hugging Face bundle (index 2 = 0.1 is unused).
EXPECTED_INDICES = {
    2: [0, 1, 3, 4, 5, 7, 10, 13, 15],
    4: [6, 8, 9, 11, 12, 14, 16, 17, 18],
}


def test_published_architecture():
    cfg = parse_transformer_config(published_transformer_config())
    assert (cfg.model_dim, cfg.ff_dim, cfg.num_visual_blocks) == (2560, 10240, 50)
    assert (cfg.head_dim, cfg.num_heads) == (128, 20)
    assert cfg.patch_size == (1, 2, 2) and cfg.axes_dims == (32, 48, 48)
    assert cfg.tile_size == 1024 and cfg.num_tokens == 4096 and cfg.token_grid == (1, 64, 64)
    assert len(cfg.baked_timesteps) == 19 and 0.1 in cfg.baked_timesteps


@pytest.mark.parametrize("factor", [2, 4])
def test_creativity_presets_map_to_published_rows(factor):
    cfg = parse_transformer_config(published_transformer_config())
    indices = [cfg.timestep_index(resolve_timestep(factor, level)) for level in range(9)]
    assert indices == EXPECTED_INDICES[factor]
    # Level 4 is the official default timestep of each factor.
    assert resolve_timestep(factor, 4) == published_model_index()["one_step_t_by_factor"][str(factor)]


def test_official_creativity_values():
    assert CREATIVITY_TIMESTEPS[2] == (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30)
    assert CREATIVITY_TIMESTEPS[4] == (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38)


def test_lookup_is_independent_of_reduced_precision():
    import torch

    cfg = parse_transformer_config(published_transformer_config())
    lossy = 0
    for t in cfg.baked_timesteps:
        assert cfg.baked_timesteps[cfg.timestep_index(t)] == t
        # The same value routed through a BF16 tensor would not match exactly; the
        # lookup works on the configuration's Python floats and never sees tensors.
        lossy += abs(float(torch.tensor(t * 1000, dtype=torch.bfloat16)) / 1000 - t) > 1e-6
    assert lossy > 0
    with pytest.raises(BundleError, match="not baked"):
        cfg.timestep_index(0.2)
    with pytest.raises(BundleError, match="not baked"):
        cfg.timestep_index(0.244 + 2e-6)
    assert cfg.timestep_index(0.244 + 5e-7) == 12


@pytest.mark.parametrize("bad", [-1, 9, 4.0, True, "4", None])
def test_invalid_creativity_rejected(bad):
    with pytest.raises(ValueError, match="creativity"):
        resolve_timestep(2, bad)


@pytest.mark.parametrize("factor", [1, 3, 8, "2"])
def test_invalid_factor_rejected(factor):
    with pytest.raises(ValueError, match="upscale_factor"):
        resolve_timestep(factor, 4)


def _mutated(**changes):
    cfg = copy.deepcopy(published_transformer_config())
    for key, value in changes.items():
        if value is KeyError:
            cfg.pop(key)
        else:
            cfg[key] = value
    return cfg


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"_class_name": "Kandinsky5Transformer3DModel"}, "unsupported transformer class"),
        ({"attention_type": "nabla"}, "regular"),
        ({"visual_cond": True}, "visual_cond"),
        ({"patch_size": [1, 4, 4]}, "patch_size"),
        ({"patch_size": [1, 2]}, "patch_size"),
        ({"axes_dims": [32, 48, 47]}, "axes_dims"),
        ({"in_visual_dim": 4}, "latent channels"),
        ({"model_dim": 2500}, "divisible"),
        ({"model_dim": KeyError}, "model_dim"),
        ({"ff_dim": -1}, "ff_dim"),
        ({"num_visual_blocks": 1.5}, "num_visual_blocks"),
        ({"baked_latent_size": 127}, "even"),
        ({"baked_latent_size": 512}, "RoPE"),
        ({"baked_timesteps": []}, "baked_timesteps"),
        ({"baked_timesteps": [0.07, 0.07]}, "increasing"),
        ({"baked_timesteps": [0.3, 0.07]}, "increasing"),
        ({"baked_timesteps": [0.07, 1.5]}, r"\(0, 1\)"),
        ({"baked_timesteps": [0.07, 0.086, 0.118]}, "creativity"),
        ({"baked_timesteps": "0.07"}, "baked_timesteps"),
    ],
)
def test_malformed_transformer_config_rejected(changes, message):
    with pytest.raises(BundleError, match=message):
        parse_transformer_config(_mutated(**changes))


def test_vae_config():
    assert validate_vae_config(published_vae_config()) == pytest.approx(0.3611)
    for key, value, message in [
        ("latent_channels", 4, "latent channels"),
        ("_class_name", "AutoencoderKLWan", "AutoencoderKL"),
        ("scaling_factor", 0, "scaling_factor"),
        ("use_quant_conv", True, "quant"),
        ("block_out_channels", [128, 256, 512], "four"),
    ]:
        cfg = published_vae_config()
        cfg[key] = value
        with pytest.raises(BundleError, match=message):
            validate_vae_config(cfg)


def test_pipeline_config():
    tcfg = parse_transformer_config(published_transformer_config())
    validate_pipeline_config(published_model_index(), tcfg)
    for key, value, message in [
        ("tile_size", 512, "tile_size"),
        ("upscale_factors", [2], "upscale_factors"),
        ("scale_factor", [1.0, 2.0, 2.0], "scale_factor"),
        ("_class_name", "FluxPipeline", "model_index"),
    ]:
        index = published_model_index()
        index[key] = value
        with pytest.raises(BundleError, match=message):
            validate_pipeline_config(index, tcfg)

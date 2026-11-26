"""Shared fixtures for building small synthetic models and bundles."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def published_transformer_config() -> dict:
    return json.loads((FIXTURES / "transformer_config.json").read_text())


def published_vae_config() -> dict:
    return json.loads((FIXTURES / "vae_config.json").read_text())


def published_model_index() -> dict:
    return json.loads((FIXTURES / "model_index.json").read_text())


def small_transformer_config(latent: int = 16, blocks: int = 2, model_dim: int = 256, ff_dim: int = 512) -> dict:
    cfg = copy.deepcopy(published_transformer_config())
    cfg.update(model_dim=model_dim, ff_dim=ff_dim, num_visual_blocks=blocks, baked_latent_size=latent, time_dim=64)
    return cfg


def random_state_dict(config, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights with realistic magnitudes; FP32-kept tensors in FP32, others BF16."""
    gen = torch.Generator().manual_seed(seed)
    state = {}
    for name, shape in config.expected_tensor_shapes().items():
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * torch.randn(shape, generator=gen)
        elif "table" in name:
            t = 0.5 * torch.randn(shape, generator=gen)
        elif name.endswith(".bias"):
            t = 0.02 * torch.randn(shape, generator=gen)
        else:
            fan_in = shape[-1]
            t = torch.randn(shape, generator=gen) / fan_in**0.5
        state[name] = t.float() if config.is_fp32_tensor(name) else t.to(torch.bfloat16)
    return state


def reference_model(config_dict: dict, state: dict[str, torch.Tensor]):
    """Upstream oracle loaded like ``from_pretrained(dtype=bfloat16)``."""
    from tests.reference_impl import KandinskyImageSRBakedTransformer3DModel

    kwargs = {k: v for k, v in config_dict.items() if not k.startswith("_")}
    model = KandinskyImageSRBakedTransformer3DModel(**kwargs)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    for name, param in model.named_parameters():
        if not any(key in name for key in model._keep_in_fp32_modules):
            param.data = param.data.to(torch.bfloat16)
    return model.eval()


def write_bundle(
    root: Path,
    transformer_cfg: dict | None = None,
    vae_cfg: dict | None = None,
    *,
    shards: int = 2,
    seed: int = 0,
    tile_size: int | None = None,
) -> Path:
    """Write a synthetic bundle with the published file layout and random weights."""
    from safetensors.torch import save_file

    from kandinsky_sr.config import parse_transformer_config
    from kandinsky_sr.loader import expected_vae_shapes

    transformer_cfg = transformer_cfg or small_transformer_config()
    config = parse_transformer_config(transformer_cfg)
    vae_cfg = vae_cfg or published_vae_config()
    root.mkdir(parents=True, exist_ok=True)
    (root / "transformer").mkdir(exist_ok=True)
    (root / "vae").mkdir(exist_ok=True)
    index = published_model_index()
    index["tile_size"] = tile_size or config.tile_size
    (root / "model_index.json").write_text(json.dumps(index))
    (root / "transformer" / "config.json").write_text(json.dumps(transformer_cfg))
    (root / "vae" / "config.json").write_text(json.dumps(vae_cfg))

    state = random_state_dict(config, seed)
    names = sorted(state)
    weight_map, total = {}, 0
    for i in range(shards):
        part = names[i::shards]
        file = f"diffusion_pytorch_model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: state[n].contiguous() for n in part}, str(root / "transformer" / file), metadata={"format": "pt"})
        for n in part:
            weight_map[n] = file
            total += state[n].numel() * state[n].element_size()
    (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    gen = torch.Generator().manual_seed(seed + 1)
    vae_state = {}
    for name, shape in expected_vae_shapes(vae_cfg).items():
        if name.endswith(("norm1.weight", "norm2.weight", "group_norm.weight", "conv_norm_out.weight")):
            t = torch.ones(shape)
        elif name.endswith(".bias"):
            t = torch.zeros(shape)
        else:
            fan_in = 1
            for d in shape[1:]:
                fan_in *= d
            t = torch.randn(shape, generator=gen) / fan_in**0.5
        vae_state[name] = t.to(torch.bfloat16)
    save_file(vae_state, str(root / "vae" / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    return root


def tiny_vae_config() -> dict:
    cfg = published_vae_config()
    cfg["block_out_channels"] = [32, 32, 32, 32]
    return cfg

"""Bundle validation, weight loading and model discovery."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kandinsky_sr.config import BundleError, parse_transformer_config
from kandinsky_sr.loader import (
    discover_bundles,
    expected_vae_shapes,
    inspect_bundle,
    load_transformer_state_dict,
    read_safetensors_header,
    validate_tables,
)
from tests.helpers import (
    FIXTURES,
    published_transformer_config,
    published_vae_config,
    random_state_dict,
    small_transformer_config,
    tiny_vae_config,
    write_bundle,
)

INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "Kandinsky-ImageSR-2.0", vae_cfg=tiny_vae_config())


def test_expected_layout_matches_published_checkpoint():
    """Names, shapes and dtypes derived from the configs equal the real Hugging Face shards."""
    layout = json.loads((FIXTURES / "published_tensor_layout.json").read_text())
    cfg = parse_transformer_config(published_transformer_config())
    published = {k: tuple(v[1]) for k, v in layout["transformer"].items()}
    assert published == cfg.expected_tensor_shapes()
    for name, (dtype, _, _) in layout["transformer"].items():
        assert (dtype == "F32") == cfg.is_fp32_tensor(name), name
    vae = {k: tuple(v[1]) for k, v in layout["vae"].items()}
    assert vae == expected_vae_shapes(published_vae_config())


def test_valid_bundle(bundle):
    info = inspect_bundle(bundle)
    assert info.vae_scaling_factor == pytest.approx(0.3611)
    assert set(info.tensors) == set(info.transformer_config.expected_tensor_shapes())
    state = load_transformer_state_dict(info, torch.bfloat16)
    expected = random_state_dict(info.transformer_config, 0)
    for name, tensor in state.items():
        assert torch.equal(tensor, expected[name]), name
        assert tensor.dtype == (torch.float32 if info.transformer_config.is_fp32_tensor(name) else torch.bfloat16)
    validate_tables(state, info.transformer_config)


def test_fp32_load_keeps_values(bundle):
    info = inspect_bundle(bundle)
    state = load_transformer_state_dict(info, torch.float32)
    assert all(t.dtype == torch.float32 for t in state.values())


def _edit_json(path: Path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_missing_shard(bundle):
    next((bundle / "transformer").glob("*-00002-of-*.safetensors")).unlink()
    with pytest.raises(BundleError, match="missing weight file"):
        inspect_bundle(bundle)


def test_incorrect_shard_mapping(bundle):
    def swap(index):
        names = sorted(index["weight_map"])
        a, b = names[0], next(n for n in names if index["weight_map"][n] != index["weight_map"][names[0]])
        index["weight_map"][a], index["weight_map"][b] = index["weight_map"][b], index["weight_map"][a]

    _edit_json(bundle / INDEX, swap)
    with pytest.raises(BundleError, match="does not match the index"):
        inspect_bundle(bundle)


def test_missing_key_in_index(bundle):
    _edit_json(bundle / INDEX, lambda i: i["weight_map"].pop("time_embed_table"))
    with pytest.raises(BundleError, match="missing"):
        inspect_bundle(bundle)


def test_unexpected_key(bundle):
    shard = next((bundle / "transformer").glob("*-00001-of-*.safetensors"))
    from safetensors.torch import load_file

    state = load_file(str(shard))
    state["visual_transformer_blocks.0.cross_attention.to_key.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(state, str(shard))
    _edit_json(
        bundle / INDEX,
        lambda i: i["weight_map"].__setitem__("visual_transformer_blocks.0.cross_attention.to_key.weight", shard.name),
    )
    with pytest.raises(BundleError, match="unexpected"):
        inspect_bundle(bundle)


def test_duplicate_key_across_shards(bundle):
    shards = sorted((bundle / "transformer").glob("*.safetensors"))
    from safetensors.torch import load_file

    first, second = load_file(str(shards[0])), load_file(str(shards[1]))
    dup = next(iter(first))
    second[dup] = first[dup]
    save_file(second, str(shards[1]))
    with pytest.raises(BundleError, match="does not match the index|more than one shard"):
        inspect_bundle(bundle)


def test_wrong_shape(bundle, tmp_path):
    cfg = small_transformer_config()
    cfg["ff_dim"] = 384  # config disagrees with the stored tensors
    (bundle / "transformer" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(BundleError, match="shape"):
        inspect_bundle(bundle)


def test_conditioning_tables_must_be_fp32(tmp_path):
    root = write_bundle(tmp_path / "b", vae_cfg=tiny_vae_config(), shards=1)
    shard = next((root / "transformer").glob("*.safetensors"))
    from safetensors.torch import load_file

    state = load_file(str(shard))
    state["time_embed_table"] = state["time_embed_table"].to(torch.bfloat16)
    save_file(state, str(shard))
    _edit_json(root / INDEX, lambda i: i["metadata"].pop("total_size"))
    with pytest.raises(BundleError, match="F32"):
        inspect_bundle(root)


def test_truncated_shard(bundle):
    shard = next((bundle / "transformer").glob("*-00001-of-*.safetensors"))
    data = shard.read_bytes()
    shard.write_bytes(data[: len(data) - 100])
    with pytest.raises(BundleError, match="truncated"):
        inspect_bundle(bundle)


def test_corrupt_header(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(BundleError, match="invalid safetensors header"):
        read_safetensors_header(path)
    path.write_bytes(b"abc")
    with pytest.raises(BundleError, match="too short"):
        read_safetensors_header(path)


def test_total_size_mismatch(bundle):
    _edit_json(bundle / INDEX, lambda i: i["metadata"].__setitem__("total_size", 1))
    with pytest.raises(BundleError, match="declares"):
        inspect_bundle(bundle)


def test_malformed_and_missing_configs(bundle):
    (bundle / "transformer" / "config.json").write_text("{not json")
    with pytest.raises(BundleError, match="cannot read"):
        inspect_bundle(bundle)
    (bundle / "transformer" / "config.json").unlink()
    with pytest.raises(BundleError, match="missing required file"):
        inspect_bundle(bundle)


def test_unsupported_architecture(bundle):
    cfg = small_transformer_config()
    cfg["_class_name"] = "Kandinsky5Transformer3DModel"
    (bundle / "transformer" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(BundleError, match="not interchangeable"):
        inspect_bundle(bundle)


def test_wrong_component_in_model_index(bundle):
    _edit_json(
        bundle / "model_index.json",
        lambda i: i.__setitem__("transformer", ["diffusers", "Kandinsky5Transformer3DModel"]),
    )
    with pytest.raises(BundleError, match="component"):
        inspect_bundle(bundle)


def test_vae_missing_tensor(bundle):
    from safetensors.torch import load_file

    path = bundle / "vae" / "diffusion_pytorch_model.safetensors"
    state = load_file(str(path))
    state.pop("decoder.conv_out.weight")
    save_file(state, str(path))
    with pytest.raises(BundleError, match="VAE weights"):
        inspect_bundle(bundle)


def test_invalid_tables(bundle):
    info = inspect_bundle(bundle)
    state = load_transformer_state_dict(info, torch.bfloat16)
    bad = dict(state)
    bad["visual_transformer_blocks.1.xattn_bias_table"] = state["visual_transformer_blocks.1.xattn_bias_table"].clone()
    bad["visual_transformer_blocks.1.xattn_bias_table"][3, 0] = float("nan")
    with pytest.raises(BundleError, match="non-finite"):
        validate_tables(bad, info.transformer_config)
    dup = dict(state)
    table = state["time_embed_table"].clone()
    table[4] = table[5]
    dup["time_embed_table"] = table
    with pytest.raises(BundleError, match="duplicate rows"):
        validate_tables(dup, info.transformer_config)


def test_discovery_across_roots(tmp_path):
    a, b = tmp_path / "unet", tmp_path / "diffusion_models"
    write_bundle(b / "Kandinsky-ImageSR-2.0", vae_cfg=tiny_vae_config())
    write_bundle(b / "kandinsky" / "ImageSR-v2", vae_cfg=tiny_vae_config())
    write_bundle(a / "Kandinsky-ImageSR-2.0", vae_cfg=tiny_vae_config(), seed=7)
    (b / "flux1-dev.safetensors").write_bytes(b"0")
    (b / ".hidden" / "x").mkdir(parents=True)
    shutil.copytree(b / "Kandinsky-ImageSR-2.0", b / ".hidden" / "x" / "Kandinsky")
    (b / "too" / "deep" / "bundle").mkdir(parents=True)
    found = discover_bundles([str(a), str(b), str(tmp_path / "missing")])
    assert list(found) == ["Kandinsky-ImageSR-2.0", "kandinsky/ImageSR-v2"]
    assert found["Kandinsky-ImageSR-2.0"] == a / "Kandinsky-ImageSR-2.0"  # first root wins
    assert discover_bundles([]) == {}

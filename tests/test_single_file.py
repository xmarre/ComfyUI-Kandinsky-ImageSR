"""Single-file checkpoint format: validation, equivalence with the bundle, conversion."""

from __future__ import annotations

import json

import pytest
import torch

from kandinsky_sr.config import BundleError
from kandinsky_sr.loader import (
    inspect_bundle,
    inspect_checkpoint,
    inspect_single_file,
    is_single_file_checkpoint,
    load_transformer_state_dict,
)
from tests.helpers import converter, rewrite_safetensors, tiny_vae_config, write_bundle, write_single_file


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle", vae_cfg=tiny_vae_config(), shards=3)


@pytest.fixture
def single(tmp_path, bundle):
    return write_single_file(tmp_path / "ckpt" / "kandinsky_imagesr_2.0.safetensors", bundle)


def test_valid_single_file_and_equivalence_with_bundle(bundle, single):
    a, b = inspect_bundle(bundle), inspect_single_file(single)
    assert b.kind == "single_file" and b.vae_path is None and a.vae_path is not None
    assert a.transformer_config == b.transformer_config and a.vae_scaling_factor == b.vae_scaling_factor
    assert {k: (v.dtype, v.shape) for k, v in a.tensors.items()} == {
        k: (v.dtype, v.shape) for k, v in b.tensors.items()
    }
    sa = load_transformer_state_dict(a, torch.bfloat16)
    sb = load_transformer_state_dict(b, torch.bfloat16)
    assert sa.keys() == sb.keys()
    for name in sa:
        assert sa[name].dtype == sb[name].dtype and torch.equal(sa[name], sb[name]), name
    assert inspect_checkpoint(single).kind == "single_file" and inspect_checkpoint(bundle).kind == "bundle"


def test_converter_verification_and_compare_detect_changes(tmp_path, bundle, single):
    conv = converter()
    assert conv.compare(single, bundle) == []
    fresh = tmp_path / "fresh.safetensors"
    records = conv.convert(bundle, fresh)
    conv.verify(records, fresh)

    # flip one byte inside the first (FP32) tensor
    def flip(data):
        out = bytearray(data)
        out[5] ^= 0xFF
        return bytes(out)

    rewrite_safetensors(single, edit_data=flip)
    problems = conv.compare(single, bundle)
    assert len(problems) == 1 and "bytes differ" in problems[0]
    rewrite_safetensors(fresh, edit_data=flip)
    with pytest.raises(SystemExit, match="byte mismatch"):
        conv.verify(records, fresh)


def test_conversion_is_deterministic(tmp_path, bundle):
    a = write_single_file(tmp_path / "a.safetensors", bundle)
    b = write_single_file(tmp_path / "b.safetensors", bundle)
    assert a.read_bytes() == b.read_bytes()


def test_converter_refuses_invalid_bundle(tmp_path, bundle):
    (bundle / "transformer" / "config.json").write_text("{}")
    with pytest.raises(BundleError):
        converter().convert(bundle, tmp_path / "out.safetensors")
    assert not (tmp_path / "out.safetensors").exists() and not (tmp_path / "out.safetensors.part").exists()


def _meta(edit):
    def apply(header):
        edit(header["__metadata__"])

    return apply


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda m: m.__setitem__("format", "something_else"), "not a Kandinsky ImageSR single-file"),
        (lambda m: m.pop("format"), "not a Kandinsky ImageSR single-file"),
        (lambda m: m.__setitem__("format_version", "2"), "format_version '2'"),
        (lambda m: m.pop("format_version"), "format_version None"),
        (lambda m: m.pop("transformer_config"), "lacks 'transformer_config'"),
        (lambda m: m.__setitem__("vae_config", "{not json"), "not valid JSON"),
        (lambda m: m.__setitem__("model_index", "[1, 2]"), "JSON object"),
        (
            lambda m: m.__setitem__(
                "transformer_config", json.dumps({**json.loads(m["transformer_config"]), "visual_cond": True})
            ),
            "visual_cond",
        ),
        (
            lambda m: m.__setitem__("model_index", json.dumps({**json.loads(m["model_index"]), "tile_size": 512})),
            "tile_size",
        ),
    ],
)
def test_metadata_validation(single, edit, message):
    rewrite_safetensors(single, edit_header=_meta(edit))
    with pytest.raises(BundleError, match=message):
        inspect_single_file(single)


def _drop_tensor(name):
    def edit(header):
        header.pop(name)

    return edit


def test_missing_tensor_leaves_gap(single):
    rewrite_safetensors(single, edit_header=_drop_tensor("time_embed_table"))
    with pytest.raises(BundleError, match="gap|missing"):
        inspect_single_file(single)


def test_truncated_and_trailing_bytes(single):
    raw = single.read_bytes()
    single.write_bytes(raw[:-64])
    with pytest.raises(BundleError, match="truncated"):
        inspect_single_file(single)
    single.write_bytes(raw + b"\0" * 64)
    with pytest.raises(BundleError, match="trailing"):
        inspect_single_file(single)


def test_overlapping_offsets(single):
    def overlap(header):
        names = sorted((k for k in header if k != "__metadata__"), key=lambda k: header[k]["data_offsets"][0])
        a, b = names[0], names[1]
        size = header[b]["data_offsets"][1] - header[b]["data_offsets"][0]
        header[b]["data_offsets"] = [header[a]["data_offsets"][0], header[a]["data_offsets"][0] + size]

    rewrite_safetensors(single, edit_header=overlap)
    with pytest.raises(BundleError, match="overlap|gap|inconsistent"):
        inspect_single_file(single)


def test_duplicate_header_entries(single):
    raw = single.read_bytes()
    import struct

    (length,) = struct.unpack("<Q", raw[:8])
    text = raw[8 : 8 + length].decode()
    first = text.index('"time_embed_table"')
    end = text.index("}", first) + 1
    duplicated = text[:1] + text[first:end] + "," + text[1:]
    encoded = duplicated.encode()
    encoded += b" " * (-len(encoded) % 8)
    single.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length :])
    with pytest.raises(BundleError, match="duplicate"):
        inspect_single_file(single)


def test_wrong_storage_dtype_and_shape(single):
    def table_bf16(header):
        header["time_embed_table"]["dtype"] = "BF16"

    rewrite_safetensors(single, edit_header=table_bf16)
    with pytest.raises(BundleError, match="inconsistent|F32"):
        inspect_single_file(single)


def test_component_prefixed_files_are_rejected(single):
    def prefix(header):
        for name in [k for k in header if k != "__metadata__"]:
            header["transformer." + name] = header.pop(name)

    rewrite_safetensors(single, edit_header=prefix)
    with pytest.raises(BundleError, match="component prefixes"):
        inspect_single_file(single)


def test_discovery_probe(tmp_path, single):
    from safetensors.torch import save_file

    other = tmp_path / "flux1-dev.safetensors"
    save_file({"x": torch.zeros(4)}, str(other), metadata={"format": "pt"})
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"\xff" * 64)
    huge = tmp_path / "huge_header.safetensors"
    huge.write_bytes((10**9).to_bytes(8, "little") + b"{}")
    assert is_single_file_checkpoint(single)
    assert not is_single_file_checkpoint(other)
    assert not is_single_file_checkpoint(junk)
    assert not is_single_file_checkpoint(huge)
    assert not is_single_file_checkpoint(tmp_path / "missing.safetensors")
    # the probe is cached per (path, size, mtime) and re-evaluated when the file changes
    rewrite_safetensors(single, edit_header=_meta(lambda m: m.__setitem__("format", "other")))
    assert not is_single_file_checkpoint(single)


def test_published_layout_fixture_matches_single_file_rules():
    """The published single-file checkpoint keeps the shard tensors' names, shapes and dtypes."""
    from kandinsky_sr.config import parse_transformer_config
    from tests.helpers import FIXTURES, published_transformer_config

    layout = json.loads((FIXTURES / "published_tensor_layout.json").read_text())["transformer"]
    cfg = parse_transformer_config(published_transformer_config())
    from kandinsky_sr.loader import _check_tensor_entries

    header = {}
    offset = 0
    item = {"F32": 4, "BF16": 2}
    for name, (dtype, shape, _) in sorted(layout.items()):
        size = item[dtype]
        for d in shape:
            size *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    tensors = {}
    total = _check_tensor_entries(header, None, cfg.expected_tensor_shapes(), cfg, tensors)
    assert total == 10250207360 and len(tensors) == 757

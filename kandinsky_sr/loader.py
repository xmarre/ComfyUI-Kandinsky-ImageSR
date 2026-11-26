"""Discovery, strict validation and memory-efficient loading of the official bundle.

Expected layout (the unmodified Hugging Face repository)::

    <bundle>/
        model_index.json
        transformer/config.json
        transformer/diffusion_pytorch_model.safetensors.index.json
        transformer/diffusion_pytorch_model-0000N-of-0000M.safetensors
        vae/config.json
        vae/diffusion_pytorch_model.safetensors

Validation reads only JSON files and safetensors headers. Every tensor name, shape
and dtype is checked against the architecture in ``transformer/config.json``; a
bundle with a missing, duplicated, unexpected, truncated or mis-shaped tensor is
rejected before any weights are read. There are no permitted extra tensors.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import (
    BundleError,
    TransformerConfig,
    load_json,
    parse_transformer_config,
    validate_pipeline_config,
    validate_vae_config,
)

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
VAE_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
_DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    shard: Path


@dataclass(frozen=True)
class BundleInfo:
    root: Path
    transformer_config: TransformerConfig
    vae_scaling_factor: float
    tensors: dict[str, TensorInfo]
    vae_path: Path


# --------------------------------------------------------------------------- safetensors headers


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors header and verify the file is not truncated."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise BundleError(f"{path} is not a safetensors file (too short)")
            (length,) = struct.unpack("<Q", prefix)
            if length <= 0 or length > _MAX_HEADER_BYTES or 8 + length > size:
                raise BundleError(f"{path} has an invalid safetensors header")
            header = json.loads(handle.read(length))
    except FileNotFoundError as error:
        raise BundleError(f"missing weight file: {path}") from error
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BundleError(f"cannot read safetensors header of {path}: {error}") from error
    if not isinstance(header, dict):
        raise BundleError(f"{path} has an invalid safetensors header")
    header.pop("__metadata__", None)
    data_end = 0
    for name, entry in header.items():
        try:
            start, end = (int(v) for v in entry["data_offsets"])
            shape = tuple(int(v) for v in entry["shape"])
            dtype = entry["dtype"]
        except (KeyError, TypeError, ValueError) as error:
            raise BundleError(f"{path}: malformed header entry for {name!r}") from error
        item = _DTYPE_BYTES.get(dtype)
        if item is not None:
            count = 1
            for dim in shape:
                count *= dim
            if end - start != count * item:
                raise BundleError(f"{path}: tensor {name!r} has inconsistent size")
        data_end = max(data_end, end)
    if 8 + length + data_end > size:
        raise BundleError(f"{path} is truncated (expected at least {8 + length + data_end} bytes, found {size})")
    return header


# --------------------------------------------------------------------------- VAE layout


def expected_vae_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Diffusers ``AutoencoderKL`` tensor names/shapes for a FLUX-style VAE config."""
    ch = [int(c) for c in cfg["block_out_channels"]]
    layers = int(cfg.get("layers_per_block", 2))
    latent = int(cfg["latent_channels"])
    shapes: dict[str, tuple[int, ...]] = {}

    def conv(name: str, out_c: int, in_c: int, k: int = 3) -> None:
        shapes[name + ".weight"] = (out_c, in_c, k, k)
        shapes[name + ".bias"] = (out_c,)

    def norm(name: str, c: int) -> None:
        shapes[name + ".weight"] = (c,)
        shapes[name + ".bias"] = (c,)

    def resnet(prefix: str, in_c: int, out_c: int) -> None:
        norm(prefix + ".norm1", in_c)
        conv(prefix + ".conv1", out_c, in_c)
        norm(prefix + ".norm2", out_c)
        conv(prefix + ".conv2", out_c, out_c)
        if in_c != out_c:
            conv(prefix + ".conv_shortcut", out_c, in_c, 1)

    def mid(prefix: str, c: int) -> None:
        resnet(prefix + ".resnets.0", c, c)
        resnet(prefix + ".resnets.1", c, c)
        norm(prefix + ".attentions.0.group_norm", c)
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            shapes[f"{prefix}.attentions.0.{proj}.weight"] = (c, c)
            shapes[f"{prefix}.attentions.0.{proj}.bias"] = (c,)

    conv("encoder.conv_in", ch[0], 3)
    prev = ch[0]
    for i, out_c in enumerate(ch):
        for j in range(layers):
            resnet(f"encoder.down_blocks.{i}.resnets.{j}", prev if j == 0 else out_c, out_c)
        prev = out_c
        if i < len(ch) - 1:
            conv(f"encoder.down_blocks.{i}.downsamplers.0.conv", out_c, out_c)
    mid("encoder.mid_block", ch[-1])
    norm("encoder.conv_norm_out", ch[-1])
    conv("encoder.conv_out", 2 * latent, ch[-1])

    rev = list(reversed(ch))
    conv("decoder.conv_in", rev[0], latent)
    mid("decoder.mid_block", rev[0])
    prev = rev[0]
    for i, out_c in enumerate(rev):
        for j in range(layers + 1):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", prev if j == 0 else out_c, out_c)
        prev = out_c
        if i < len(rev) - 1:
            conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", out_c, out_c)
    norm("decoder.conv_norm_out", rev[-1])
    conv("decoder.conv_out", 3, rev[-1])
    return shapes


# --------------------------------------------------------------------------- validation


def _check_component(index: dict[str, Any], name: str, expected: list[str]) -> None:
    value = index.get(name)
    if value != expected:
        raise BundleError(f"model_index.json component {name!r} is {value!r}, expected {expected!r}")


def inspect_bundle(root: str | os.PathLike[str]) -> BundleInfo:
    """Validate a bundle directory without reading any tensor data."""
    root = Path(root)
    if not root.is_dir():
        raise BundleError(f"model bundle directory not found: {root}")

    transformer_config = parse_transformer_config(load_json(root / "transformer" / "config.json"))
    model_index = load_json(root / "model_index.json")
    validate_pipeline_config(model_index, transformer_config)
    _check_component(model_index, "transformer", ["kandinsky_image_sr", "KandinskyImageSRBakedTransformer3DModel"])
    _check_component(model_index, "vae", ["diffusers", "AutoencoderKL"])

    vae_cfg = load_json(root / "vae" / "config.json")
    scaling = validate_vae_config(vae_cfg)

    tensors = _validate_transformer_shards(root / "transformer", transformer_config)

    vae_path = root / "vae" / VAE_WEIGHTS_NAME
    vae_header = read_safetensors_header(vae_path)
    expected_vae = expected_vae_shapes(vae_cfg)
    _compare_names(set(vae_header), set(expected_vae), "VAE")
    for name, shape in expected_vae.items():
        entry = vae_header[name]
        if tuple(entry["shape"]) != shape:
            raise BundleError(f"VAE tensor {name!r} has shape {entry['shape']}, expected {list(shape)}")
        if entry["dtype"] not in _DTYPES:
            raise BundleError(f"VAE tensor {name!r} has unsupported dtype {entry['dtype']}")

    return BundleInfo(root, transformer_config, scaling, tensors, vae_path)


def _compare_names(found: set[str], expected: set[str], what: str) -> None:
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing (e.g. {missing[:3]})")
        if unexpected:
            parts.append(f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        raise BundleError(f"{what} weights do not match the expected architecture: " + "; ".join(parts))


def _validate_transformer_shards(directory: Path, config: TransformerConfig) -> dict[str, TensorInfo]:
    index = load_json(directory / INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise BundleError(f"{directory / INDEX_NAME} has no weight_map")
    expected = config.expected_tensor_shapes()
    _compare_names(set(weight_map), set(expected), "transformer index")

    shard_names = sorted(set(weight_map.values()))
    for shard in shard_names:
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise BundleError(f"invalid shard file name in index: {shard!r}")

    tensors: dict[str, TensorInfo] = {}
    total_bytes = 0
    for shard in shard_names:
        path = directory / shard
        header = read_safetensors_header(path)
        mapped = {name for name, file in weight_map.items() if file == shard}
        present = set(header)
        if mapped != present:
            missing = sorted(mapped - present)
            extra = sorted(present - mapped)
            raise BundleError(f"shard {shard} does not match the index: missing {missing[:3]}, unexpected {extra[:3]}")
        for name in present:
            if name in tensors:
                raise BundleError(f"tensor {name!r} appears in more than one shard")
            entry = header[name]
            shape = tuple(int(v) for v in entry["shape"])
            dtype = entry["dtype"]
            if shape != expected[name]:
                raise BundleError(f"tensor {name!r} has shape {list(shape)}, expected {list(expected[name])}")
            if config.is_fp32_tensor(name):
                if dtype != "F32":
                    raise BundleError(f"tensor {name!r} must be stored in F32 (found {dtype})")
            elif dtype not in ("BF16", "F32"):
                raise BundleError(f"tensor {name!r} has unsupported dtype {dtype}")
            tensors[name] = TensorInfo(dtype, shape, path)
            total_bytes += int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])

    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata and int(metadata["total_size"]) != total_bytes:
        raise BundleError(
            f"transformer shards hold {total_bytes} bytes but the index declares {metadata['total_size']}"
        )
    return tensors


# --------------------------------------------------------------------------- weight loading


def load_transformer_state_dict(bundle: BundleInfo, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read all transformer tensors to CPU.

    Tensors the reference keeps in FP32 stay FP32. Every other tensor is converted
    to ``dtype`` one at a time, so peak host memory stays close to one copy of the
    model in its final precision.
    """
    from safetensors import safe_open

    config = bundle.transformer_config
    by_shard: dict[Path, list[str]] = {}
    for name, info in bundle.tensors.items():
        by_shard.setdefault(info.shard, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(by_shard):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in sorted(by_shard[shard]):
                tensor = handle.get_tensor(name)
                target = torch.float32 if config.is_fp32_tensor(name) else dtype
                if tensor.dtype != target:
                    tensor = tensor.to(target)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                state[name] = tensor
    if set(state) != set(config.expected_tensor_shapes()):
        raise BundleError("internal error: incomplete transformer state dict")
    return state


def validate_tables(state: dict[str, torch.Tensor], config: TransformerConfig) -> None:
    """Sanity-check the baked conditioning tables (finite, non-degenerate)."""
    tables = ["time_embed_table"] + [
        f"visual_transformer_blocks.{i}.xattn_bias_table" for i in range(config.num_visual_blocks)
    ]
    for name in tables:
        table = state[name]
        if not bool(torch.isfinite(table).all()):
            raise BundleError(f"baked table {name!r} contains non-finite values")
    time_table = state["time_embed_table"]
    rows = time_table.shape[0]
    if rows > 1:
        distinct = torch.cdist(time_table.double(), time_table.double())
        distinct.fill_diagonal_(float("inf"))
        if bool((distinct.min(dim=1).values == 0).any()):
            raise BundleError("time_embed_table contains duplicate rows; baked timesteps are not distinct")


# --------------------------------------------------------------------------- discovery


def is_bundle_dir(path: Path) -> bool:
    return (path / "model_index.json").is_file() and (path / "transformer" / "config.json").is_file()


def discover_bundles(search_roots: list[str]) -> dict[str, Path]:
    """Find bundles directly inside each root or one directory level deeper.

    Only directory entries are inspected (no recursive walk, no file reads beyond
    ``stat``), so large model folders are cheap to scan. Names are relative paths
    with forward slashes; the first root wins on name collisions, matching
    ComfyUI's folder precedence.
    """
    found: dict[str, Path] = {}
    for root in search_roots:
        base = Path(root)
        if not base.is_dir():
            continue
        candidates: list[tuple[str, Path]] = []
        try:
            children = sorted(
                (p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name
            )
        except OSError:
            continue
        for child in children:
            if is_bundle_dir(child):
                candidates.append((child.name, child))
                continue
            try:
                grandchildren = sorted(
                    (p for p in child.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name
                )
            except OSError:
                continue
            for grandchild in grandchildren:
                if is_bundle_dir(grandchild):
                    candidates.append((f"{child.name}/{grandchild.name}", grandchild))
        for name, path in candidates:
            found.setdefault(name, path)
    return dict(sorted(found.items()))

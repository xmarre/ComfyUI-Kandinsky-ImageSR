"""Discovery, strict validation and memory-efficient loading of model checkpoints.

Two equivalent representations of the transformer are supported.

Single-file checkpoint (preferred): one ``.safetensors`` file with the
transformer tensors under their original names, shapes, dtypes and bytes, and
safetensors metadata ``format = kandinsky_image_sr_single_file``,
``format_version = 1`` and the bundle's ``transformer_config``, ``vae_config`` and
``model_index`` as JSON strings. It holds no VAE; the FLUX.1 VAE is loaded
separately.

Official Hugging Face bundle (the unmodified repository)::

    <bundle>/
        model_index.json
        transformer/config.json
        transformer/diffusion_pytorch_model.safetensors.index.json
        transformer/diffusion_pytorch_model-0000N-of-0000M.safetensors
        vae/config.json
        vae/diffusion_pytorch_model.safetensors

Validation reads only JSON and safetensors headers. Both forms go through the
same configuration parser and the same tensor checks: every tensor name, shape
and storage dtype is compared with the architecture, and a checkpoint with a
missing, duplicated, unexpected, overlapping, truncated or mis-shaped tensor is
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
SINGLE_FILE_FORMAT = "kandinsky_image_sr_single_file"
SINGLE_FILE_FORMAT_VERSION = "1"
#: Headers larger than this are not inspected during discovery (not a candidate).
_DISCOVERY_MAX_HEADER_BYTES = 16 * 1024 * 1024
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
    """A validated checkpoint: official bundle directory or single-file checkpoint."""

    root: Path
    transformer_config: TransformerConfig
    vae_scaling_factor: float
    tensors: dict[str, TensorInfo]
    #: Bundled VAE weights (official bundle only); ``None`` for single-file checkpoints.
    vae_path: Path | None
    kind: str = "bundle"


# --------------------------------------------------------------------------- safetensors headers


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate entry {key!r} in safetensors header")
        result[key] = value
    return result


def read_safetensors(path: Path, max_header: int = _MAX_HEADER_BYTES) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and check a safetensors header; returns ``(tensor entries, metadata)``.

    Verifies that no name is duplicated, every tensor's byte range matches its
    shape and dtype, the ranges tile the data section without gaps or overlaps, and
    the file is neither truncated nor followed by stray data.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise BundleError(f"{path} is not a safetensors file (too short)")
            (length,) = struct.unpack("<Q", prefix)
            if length <= 0 or length > max_header or 8 + length > size:
                raise BundleError(f"{path} has an invalid safetensors header")
            header = json.loads(handle.read(length), object_pairs_hook=_reject_duplicates)
    except FileNotFoundError as error:
        raise BundleError(f"missing weight file: {path}") from error
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BundleError(f"cannot read safetensors header of {path}: {error}") from error
    if not isinstance(header, dict):
        raise BundleError(f"{path} has an invalid safetensors header")
    metadata = header.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise BundleError(f"{path} has malformed safetensors metadata")
    ranges = []
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
        ranges.append((start, end, name))
    position = 0
    for start, end, name in sorted(ranges):
        if start != position or end < start:
            raise BundleError(f"{path}: tensor {name!r} overlaps another tensor or leaves a gap")
        position = end
    if 8 + length + position > size:
        raise BundleError(f"{path} is truncated (expected {8 + length + position} bytes, found {size})")
    if 8 + length + position < size:
        raise BundleError(f"{path} has {size - 8 - length - position} unexpected trailing bytes")
    return header, metadata


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Tensor entries of a checked safetensors header (see :func:`read_safetensors`)."""
    return read_safetensors(path)[0]


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
        total_bytes += _check_tensor_entries(header, path, expected, config, tensors)

    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata and int(metadata["total_size"]) != total_bytes:
        raise BundleError(
            f"transformer shards hold {total_bytes} bytes but the index declares {metadata['total_size']}"
        )
    return tensors


def _check_tensor_entries(
    header: dict[str, Any],
    path: Path,
    expected: dict[str, tuple[int, ...]],
    config: TransformerConfig,
    tensors: dict[str, TensorInfo],
) -> int:
    """Check shapes and storage dtypes of ``header`` and record them; returns the byte count."""
    total = 0
    for name in sorted(header):
        entry = header[name]
        if name not in expected:
            raise BundleError(f"unexpected transformer tensor {name!r}")
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
        total += int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])
    return total


# --------------------------------------------------------------------------- single-file checkpoints


def _metadata_json(metadata: dict[str, str], key: str, path: Path) -> dict[str, Any]:
    if key not in metadata:
        raise BundleError(f"{path}: single-file metadata lacks {key!r}")
    try:
        value = json.loads(metadata[key])
    except json.JSONDecodeError as error:
        raise BundleError(f"{path}: metadata {key!r} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path}: metadata {key!r} must be a JSON object")
    return value


def inspect_single_file(path: str | os.PathLike[str]) -> BundleInfo:
    """Validate a single-file checkpoint without reading any tensor data."""
    path = Path(path)
    if not path.is_file():
        raise BundleError(f"checkpoint file not found: {path}")
    header, metadata = read_safetensors(path)
    found = metadata.get("format")
    if found != SINGLE_FILE_FORMAT:
        raise BundleError(
            f"{path.name} is not a Kandinsky ImageSR single-file checkpoint "
            f"(safetensors metadata format is {found!r}, expected {SINGLE_FILE_FORMAT!r})"
        )
    version = metadata.get("format_version")
    if version != SINGLE_FILE_FORMAT_VERSION:
        raise BundleError(
            f"{path.name} uses single-file format_version {version!r}; this version of the nodes supports "
            f"{SINGLE_FILE_FORMAT_VERSION!r}"
        )
    transformer_config = parse_transformer_config(_metadata_json(metadata, "transformer_config", path))
    model_index = _metadata_json(metadata, "model_index", path)
    validate_pipeline_config(model_index, transformer_config)
    _check_component(model_index, "transformer", ["kandinsky_image_sr", "KandinskyImageSRBakedTransformer3DModel"])
    _check_component(model_index, "vae", ["diffusers", "AutoencoderKL"])
    scaling = validate_vae_config(_metadata_json(metadata, "vae_config", path))

    expected = transformer_config.expected_tensor_shapes()
    if any(name.startswith(("transformer.", "vae.")) for name in header):
        raise BundleError(
            f"{path.name} uses component prefixes (combined transformer + VAE file); only transformer-only "
            "single-file checkpoints are supported"
        )
    _compare_names(set(header), set(expected), "single-file checkpoint")
    tensors: dict[str, TensorInfo] = {}
    _check_tensor_entries(header, path, expected, transformer_config, tensors)
    return BundleInfo(path, transformer_config, scaling, tensors, None, kind="single_file")


def inspect_checkpoint(path: str | os.PathLike[str]) -> BundleInfo:
    """Validate a bundle directory or a single-file checkpoint."""
    path = Path(path)
    return inspect_single_file(path) if path.is_file() else inspect_bundle(path)


def is_single_file_checkpoint(path: str | os.PathLike[str]) -> bool:
    """Cheap discovery test: reads only the safetensors header of ``path``.

    Files whose header is larger than a few MiB (not this format) are skipped
    without reading it. Results are cached by path, size and modification time.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return False
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _SINGLE_FILE_CACHE.get(key)
    if cached is not None:
        return cached
    result = False
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) == 8:
                (length,) = struct.unpack("<Q", prefix)
                if 0 < length <= _DISCOVERY_MAX_HEADER_BYTES and 8 + length <= stat.st_size:
                    header = json.loads(handle.read(length))
                    metadata = header.get("__metadata__") if isinstance(header, dict) else None
                    result = isinstance(metadata, dict) and metadata.get("format") == SINGLE_FILE_FORMAT
    except (OSError, ValueError):
        result = False
    if len(_SINGLE_FILE_CACHE) > 4096:
        _SINGLE_FILE_CACHE.clear()
    _SINGLE_FILE_CACHE[key] = result
    return result


_SINGLE_FILE_CACHE: dict[tuple[str, int, int], bool] = {}


# --------------------------------------------------------------------------- weight loading


def load_transformer_state_dict(
    bundle: BundleInfo, dtype: torch.dtype, modulation_dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Read all transformer tensors to CPU.

    The baked conditioning tables stay FP32. The FP32 modulation projections are
    stored as ``modulation_dtype``; every other tensor is converted to ``dtype``.
    Tensors are converted one at a time, so peak host memory stays close to one
    copy of the model in its final precision.
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
                if not config.is_fp32_tensor(name):
                    target = dtype
                elif name.endswith("_table"):
                    target = torch.float32
                else:
                    target = modulation_dtype
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

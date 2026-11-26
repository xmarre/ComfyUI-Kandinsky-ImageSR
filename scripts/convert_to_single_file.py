#!/usr/bin/env python3
"""Convert the official Kandinsky ImageSR 2.0 bundle to the single-file checkpoint.

Writes ``kandinsky_imagesr_2.0.safetensors`` in the community single-file format
(``format = kandinsky_image_sr_single_file``, ``format_version = 1``): the
transformer tensors with their original names, shapes, dtypes and bytes, and the
bundle's ``transformer/config.json``, ``vae/config.json`` and ``model_index.json``
as compact JSON strings in the safetensors metadata. The VAE is not included; use
the FLUX.1 ``ae.safetensors`` with ComfyUI's Load VAE node.

Tensor order and header layout follow the reference converter of
ComfyUI-KSR-Unofficial (``tools/merge_kandinsky_sr.py``, MIT License, Copyright (c)
2026 the-aSak), so converting the unmodified bundle reproduces the published file
byte for byte.

The bundle is validated first with this package's strict checks. Tensor data is
copied in 64 MiB blocks without loading the model; peak memory stays small.

Usage::

    # convert, then re-read and compare every tensor
    python scripts/convert_to_single_file.py convert <bundle dir> <output.safetensors> --verify
    # check an existing single-file checkpoint (e.g. a downloaded one) against a bundle
    python scripts/convert_to_single_file.py compare <checkpoint.safetensors> <bundle dir>
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky_sr.loader import (  # noqa: E402
    SINGLE_FILE_FORMAT,
    SINGLE_FILE_FORMAT_VERSION,
    inspect_bundle,
    inspect_single_file,
    read_safetensors,
)

CHUNK = 64 * 1024 * 1024
DTYPE_SIZE = {"F32": 4, "BF16": 2, "F16": 2}


def _compact(path: Path) -> str:
    # Key order is preserved; only whitespace changes.
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))


def _records(bundle: Path) -> list[dict]:
    info = inspect_bundle(bundle)  # strict validation of configs, index, shards and tensors
    records = []
    for shard in sorted({t.shard for t in info.tensors.values()}):
        with open(shard, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        header.pop("__metadata__", None)
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            records.append(
                {
                    "name": name,
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "file": shard,
                    "start": 8 + length + begin,
                    "length": end - begin,
                }
            )
    # Wider dtypes first keeps every tensor aligned to its element size.
    records.sort(key=lambda r: (-DTYPE_SIZE[r["dtype"]], r["name"]))
    return records


def convert(bundle: Path, output: Path) -> list[dict]:
    records = _records(bundle)
    metadata = {
        "format": SINGLE_FILE_FORMAT,
        "format_version": SINGLE_FILE_FORMAT_VERSION,
        "transformer_config": _compact(bundle / "transformer" / "config.json"),
        "vae_config": _compact(bundle / "vae" / "config.json"),
        "model_index": _compact(bundle / "model_index.json"),
    }
    header: dict = {"__metadata__": metadata}
    offset = 0
    for r in records:
        header[r["name"]] = {"dtype": r["dtype"], "shape": r["shape"], "data_offsets": [offset, offset + r["length"]]}
        r["out_start"] = offset
        offset += r["length"]
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    data_start = 8 + len(header_bytes)

    partial = output.with_suffix(output.suffix + ".part")
    sources: dict[Path, object] = {}
    try:
        with open(partial, "wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            for r in records:
                src = sources.get(r["file"]) or sources.setdefault(r["file"], open(r["file"], "rb"))  # noqa: SIM115  (closed in finally)
                src.seek(r["start"])
                remaining = r["length"]
                while remaining:
                    block = src.read(min(CHUNK, remaining))
                    if not block:
                        raise OSError(f"unexpected end of {r['file']}")
                    dst.write(block)
                    remaining -= len(block)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        for handle in sources.values():
            handle.close()
    partial.replace(output)
    for r in records:
        r["out_start"] += data_start
    return records


def verify(records: list[dict], output: Path) -> None:
    """Compare every tensor's bytes in ``output`` with the source shards."""
    inspect_single_file(output)  # the written file must pass the loader's own validation
    sources: dict[Path, object] = {}
    try:
        with open(output, "rb") as out:
            for r in records:
                src = sources.get(r["file"]) or sources.setdefault(r["file"], open(r["file"], "rb"))  # noqa: SIM115  (closed in finally)
                src.seek(r["start"])
                out.seek(r["out_start"])
                remaining = r["length"]
                while remaining:
                    n = min(CHUNK, remaining)
                    if src.read(n) != out.read(n):
                        raise SystemExit(f"byte mismatch in tensor {r['name']}")
                    remaining -= n
    finally:
        for handle in sources.values():
            handle.close()


def compare(checkpoint: Path, bundle: Path) -> list[str]:
    """Compare a single-file checkpoint with a bundle: metadata, tensor set, dtypes, shapes, bytes.

    Returns a list of human-readable differences (empty when identical).
    """
    inspect_single_file(checkpoint)
    problems = []
    header, metadata = read_safetensors(checkpoint)
    expected_meta = {
        "transformer_config": bundle / "transformer" / "config.json",
        "vae_config": bundle / "vae" / "config.json",
        "model_index": bundle / "model_index.json",
    }
    for key, path in expected_meta.items():
        if json.loads(metadata[key]) != json.loads(path.read_text(encoding="utf-8")):
            problems.append(f"metadata {key} differs from {path}")
    with open(checkpoint, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
    data_start = 8 + length
    records = {r["name"]: r for r in _records(bundle)}
    if set(records) != set(header):
        problems.append("tensor names differ")
        return problems
    sources: dict[Path, object] = {}
    try:
        with open(checkpoint, "rb") as out:
            for name in sorted(records):
                r, entry = records[name], header[name]
                if entry["dtype"] != r["dtype"] or list(entry["shape"]) != list(r["shape"]):
                    problems.append(
                        f"{name}: dtype/shape {entry['dtype']} {entry['shape']} vs {r['dtype']} {r['shape']}"
                    )
                    continue
                src = sources.get(r["file"]) or sources.setdefault(r["file"], open(r["file"], "rb"))  # noqa: SIM115  (closed in finally)
                src.seek(r["start"])
                out.seek(data_start + entry["data_offsets"][0])
                remaining = r["length"]
                while remaining:
                    n = min(CHUNK, remaining)
                    if src.read(n) != out.read(n):
                        problems.append(f"{name}: tensor bytes differ")
                        break
                    remaining -= n
    finally:
        for handle in sources.values():
            handle.close()
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="write a single-file checkpoint from a bundle")
    conv.add_argument("bundle", type=Path, help="local copy of kandinskylab/Kandinsky-ImageSR-2.0")
    conv.add_argument("output", type=Path, help="target .safetensors file")
    conv.add_argument("--verify", action="store_true", help="re-read and compare every tensor byte afterwards")
    comp = sub.add_parser("compare", help="compare an existing single-file checkpoint with a bundle")
    comp.add_argument("checkpoint", type=Path)
    comp.add_argument("bundle", type=Path)
    args = parser.parse_args()

    if args.command == "compare":
        problems = compare(args.checkpoint, args.bundle)
        for problem in problems:
            print(problem)
        if problems:
            return 2
        print("identical: metadata matches the bundle configs and every tensor is byte-identical")
        return 0

    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = convert(args.bundle, args.output)
    print(f"wrote {len(records)} tensors, {args.output.stat().st_size / 1e9:.2f} GB to {args.output}")
    if args.verify:
        verify(records, args.output)
        print("verified: every tensor is byte-identical to the bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())

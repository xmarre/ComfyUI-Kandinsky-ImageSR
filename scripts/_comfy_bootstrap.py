"""Import ComfyUI from a checkout for standalone scripts (no server is started)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def bootstrap(comfyui_path: str, comfy_args: list[str]) -> None:
    """Make ``comfy`` importable and apply ComfyUI command line flags (e.g. ``--lowvram``)."""
    root = Path(comfyui_path).expanduser().resolve()
    if not (root / "comfy").is_dir():
        raise SystemExit(f"{root} is not a ComfyUI checkout (no comfy/ directory)")
    for path in (str(root), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import comfy.cli_args as cli

    parsed = cli.parser.parse_args(comfy_args)
    for key, value in vars(parsed).items():
        setattr(cli.args, key, value)
    import comfy.model_management  # noqa: F401  (device selection happens at import)

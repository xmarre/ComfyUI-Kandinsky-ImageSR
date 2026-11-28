"""Test configuration.

Pure-PyTorch tests always run. Tests that need ComfyUI use the ``comfy`` fixture,
which imports ComfyUI from the ``COMFYUI_PATH`` environment variable (forced to CPU
mode) and skips with an explicit reason when it is not available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_comfy():
    path = os.environ.get("COMFYUI_PATH")
    if not path or not Path(path, "comfy").is_dir():
        return None, "COMFYUI_PATH is not set to a ComfyUI checkout"
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from comfy.cli_args import args

        args.cpu = True
        import comfy.model_management  # noqa: F401
    except Exception as error:  # pragma: no cover - environment dependent
        return None, f"cannot import ComfyUI: {error!r}"
    return path, None


_COMFY_PATH, _COMFY_REASON = _setup_comfy()


@pytest.fixture(scope="session")
def comfy():
    if _COMFY_PATH is None:
        pytest.skip(_COMFY_REASON)
    return _COMFY_PATH


def pytest_report_header(config):
    return f"ComfyUI: {_COMFY_PATH or 'unavailable (' + str(_COMFY_REASON) + ')'}"

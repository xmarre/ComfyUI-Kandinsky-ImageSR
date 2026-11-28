"""The direct-restoration study script: exact-endpoint control and resume provenance."""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from kandinsky_sr.direct import DirectGrid, plan_grid, sample_tiles
from kandinsky_sr.pipeline import TiledRestorer
from kandinsky_sr.resize import scaled_length
from tests.test_pipeline import TILE, FakeBackend, _images

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location(
        "research_direct_fractional", ROOT / "scripts" / "research_direct_fractional.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(h, w, scale):
    out = (scaled_length(h, Decimal(scale)), scaled_length(w, Decimal(scale)))
    return plan_grid(h, w, out, 2, TILE, 0.25)


def test_exact_family_size_control_uses_the_native_preprocessing(study):
    """At exactly 2x the plan is the native grid; the study must use the native tile input."""
    images = _images(1, 40, 52)
    grid = _plan(40, 52, "2")
    assert not isinstance(grid, DirectGrid) and grid.num_tiles == 1
    backend = FakeBackend()
    tile = study.model_input(backend, grid, images)
    expected = TiledRestorer(backend, grid)._model_pixels(images, [(0, 0, 0)], backend.compute_device)
    assert tile.shape == (1, 3, TILE, TILE) and torch.equal(tile, expected)


def test_fractional_control_uses_direct_sampling(study):
    images = _images(1, 40, 52)
    grid = _plan(40, 52, "1.2")
    assert isinstance(grid, DirectGrid) and grid.num_tiles == 1
    backend = FakeBackend()
    tile = study.model_input(backend, grid, images)
    assert torch.equal(tile, sample_tiles(images, [(0, 0, 0)], grid, backend.compute_device))


def test_multi_tile_plans_are_rejected(study):
    grid = _plan(150, 170, "1.3")
    assert grid.num_tiles > 1
    with pytest.raises(ValueError, match="single model tile"):
        study.model_input(FakeBackend(), grid, _images(1, 150, 170))


def _manifest(**changes):
    base = {"implementation_sha256": "a", "model": {"header_sha256": "m"}, "strengths": [0.0, 0.5, 1.0]}
    return {**base, **changes}


def test_resume_requires_an_identical_manifest(study, tmp_path):
    assert study.check_resume(tmp_path, _manifest()) == []  # fresh directory
    (tmp_path / "results.json").write_text(json.dumps([{"image": "x"}]))
    with pytest.raises(SystemExit, match="without a manifest"):
        study.check_resume(tmp_path, _manifest())
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest()))
    assert study.check_resume(tmp_path, _manifest()) == [{"image": "x"}]
    for changed, key in (
        (_manifest(strengths=[0.0, 0.25, 0.5, 1.0]), "strengths"),
        (_manifest(implementation_sha256="b"), "implementation_sha256"),
        (_manifest(model={"header_sha256": "other"}), "model"),
    ):
        with pytest.raises(SystemExit, match=key):
            study.check_resume(tmp_path, changed)


def test_implementation_digest_covers_the_package_sources(study, tmp_path):
    digest = study.implementation_digest()
    assert len(digest) == 64 and digest == study.implementation_digest()

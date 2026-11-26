"""Checks against the real published files; skipped unless their paths are given.

KANDINSKY_SR_BUNDLE       official Hugging Face bundle folder
KANDINSKY_SR_SINGLE_FILE  kandinsky_imagesr_2.0.safetensors
KANDINSKY_SR_FLUX_VAE     FLUX.1 ae.safetensors
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import torch

from kandinsky_sr.loader import inspect_bundle, inspect_single_file

BUNDLE = os.environ.get("KANDINSKY_SR_BUNDLE")
SINGLE = os.environ.get("KANDINSKY_SR_SINGLE_FILE")
FLUX_VAE = os.environ.get("KANDINSKY_SR_FLUX_VAE")

pytestmark = pytest.mark.real_model


@pytest.mark.skipif(not (BUNDLE and SINGLE), reason="set KANDINSKY_SR_BUNDLE and KANDINSKY_SR_SINGLE_FILE")
def test_single_file_matches_official_bundle_byte_for_byte():
    from tests.helpers import converter

    a, b = inspect_bundle(BUNDLE), inspect_single_file(SINGLE)
    assert a.transformer_config == b.transformer_config and a.vae_scaling_factor == b.vae_scaling_factor
    assert converter().compare(Path(SINGLE), Path(BUNDLE)) == []


@pytest.mark.skipif(not (BUNDLE and FLUX_VAE), reason="set KANDINSKY_SR_BUNDLE and KANDINSKY_SR_FLUX_VAE")
def test_bundled_vae_is_flux_ae_rounded_to_bf16(comfy, caplog):
    import comfy.diffusers_convert
    import comfy.sd
    from safetensors.torch import load_file

    from kandinsky_sr.vae import validate_imagesr_vae

    bundled = comfy.diffusers_convert.convert_vae_state_dict(
        load_file(str(Path(BUNDLE) / "vae" / "diffusion_pytorch_model.safetensors"))
    )
    flux = load_file(FLUX_VAE)
    assert bundled.keys() == flux.keys()
    assert all(torch.equal(bundled[k], flux[k].to(torch.bfloat16)) for k in flux)
    for dtype in (torch.bfloat16, torch.float32):
        vae = comfy.sd.VAE(sd=dict(flux), dtype=dtype)
        # the shipped fingerprint and BF16 digests accept the real FLUX.1 VAE without a warning
        with caplog.at_level(logging.WARNING):
            validate_imagesr_vae(vae)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        if dtype == torch.bfloat16:  # ComfyUI's bf16 load gives exactly the bundled weights
            state = vae.first_stage_model.state_dict()
            assert all(torch.equal(state[k], bundled[k]) for k in bundled)

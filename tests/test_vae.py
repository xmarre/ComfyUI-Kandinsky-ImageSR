"""ImageSR latent normalisation: scale only, applied exactly once, no FLUX shift."""

from __future__ import annotations

import pytest
import torch

from kandinsky_sr.vae import ImageSRVAE, decode_autocast

SCALE = 0.3611
SHIFT = 0.1159


class RecordingAE(torch.nn.Module):
    """Linear stand-in autoencoder that records what it receives."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def encode(self, x):
        self.seen["encode_in"] = x.clone()
        return (x[:, :1].repeat(1, 16, 1, 1)[:, :, ::8, ::8] * 0.5).to(x.dtype)

    def decode(self, z):
        self.seen["decode_in"] = z.clone()
        return (z[:, :3] * 3.0).repeat_interleave(8, -1).repeat_interleave(8, -2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encode_normalisation(dtype):
    ae = RecordingAE()
    vae = ImageSRVAE(ae, SCALE, torch.device("cpu"), dtype)
    pixels = torch.rand(2, 3, 16, 16)
    z = vae.encode(pixels)
    assert torch.equal(ae.seen["encode_in"], (pixels * 2.0 - 1.0).to(dtype))
    raw = ae.encode((pixels * 2.0 - 1.0).to(dtype))
    assert z.dtype == torch.float32
    assert torch.equal(z, raw.float() * SCALE)  # scale once, no shift
    assert not torch.allclose(z, (raw.float() - SHIFT) * SCALE)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_normalisation(dtype):
    ae = RecordingAE()
    vae = ImageSRVAE(ae, SCALE, torch.device("cpu"), dtype)
    z = torch.randn(2, 16, 2, 2)
    out = vae.decode(z)
    assert torch.equal(ae.seen["decode_in"], (z / SCALE).to(dtype))
    decoded = ae.decode((z / SCALE).to(dtype))
    assert torch.equal(out, (decoded.float().clamp(-1, 1) + 1) * 0.5)
    assert out.dtype == torch.float32 and float(out.min()) >= 0 and float(out.max()) <= 1


def test_round_trip_through_identity_autoencoder_is_lossless():
    class Identity(torch.nn.Module):
        def encode(self, x):
            return x

        def decode(self, z):
            return z

    vae = ImageSRVAE(Identity(), SCALE, torch.device("cpu"), torch.float32)
    pixels = torch.rand(1, 3, 8, 8)
    torch.testing.assert_close(vae.decode(vae.encode(pixels)), pixels, rtol=0, atol=1e-6)


def test_decode_autocast_only_on_cuda_half():
    assert not isinstance(decode_autocast(torch.device("cpu"), torch.bfloat16), torch.autocast)
    assert not isinstance(decode_autocast(torch.device("cuda"), torch.float32), torch.autocast)
    if torch.cuda.is_available():
        assert isinstance(decode_autocast(torch.device("cuda"), torch.bfloat16), torch.autocast)


@pytest.fixture(scope="module")
def comfy_vae(comfy):
    import comfy.sd
    from safetensors.torch import load_file

    from tests.helpers import published_vae_config, write_bundle

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = write_bundle(Path(tmp) / "b", vae_cfg=published_vae_config(), shards=1)
        state = load_file(str(root / "vae" / "diffusion_pytorch_model.safetensors"))
    return comfy.sd.VAE(sd=state, dtype=torch.float32)


@torch.no_grad()
def test_comfy_flux_vae_returns_raw_latents_and_imagesr_scales_once(comfy_vae):
    import comfy.latent_formats

    vae = ImageSRVAE(comfy_vae.first_stage_model, SCALE, comfy_vae.device, comfy_vae.vae_dtype)
    pixels = torch.rand(2, 3, 64, 64)
    z = vae.encode(pixels)
    raw = comfy_vae.first_stage_model.encode(pixels * 2 - 1)
    assert torch.equal(z, raw.float() * SCALE)
    # comfy.sd.VAE.encode returns the same raw latents (no scale/shift inside)
    via_comfy = comfy_vae.encode(pixels.movedim(1, -1))
    torch.testing.assert_close(via_comfy.float(), raw.float(), rtol=1e-5, atol=1e-5)
    # the FLUX latent format would additionally subtract the shift; ImageSR must not
    flux = comfy.latent_formats.Flux()
    assert flux.scale_factor == pytest.approx(SCALE) and flux.shift_factor == pytest.approx(SHIFT)
    assert not torch.allclose(z, flux.process_in(raw.float()))
    # posterior mode is deterministic
    assert torch.equal(vae.encode(pixels), z)
    # batch elements are encoded independently
    torch.testing.assert_close(vae.encode(pixels[1:]), z[1:], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_comfy_flux_vae_decode_inverts_scale(comfy_vae):
    vae = ImageSRVAE(comfy_vae.first_stage_model, SCALE, comfy_vae.device, comfy_vae.vae_dtype)
    z = torch.randn(1, 16, 8, 8)
    out = vae.decode(z)
    raw = comfy_vae.first_stage_model.decode(z / SCALE)
    assert torch.equal(out, (raw.float().clamp(-1, 1) + 1) * 0.5)
    assert out.shape == (1, 3, 64, 64)

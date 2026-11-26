"""FLUX VAE stages with the ImageSR latent normalisation.

The official pipeline multiplies the posterior mode by ``scaling_factor`` and
divides by it before decoding. It does **not** apply the FLUX ``shift_factor``.
ComfyUI's ``comfy.sd.VAE`` returns raw (unscaled, unshifted) latents from the
autoencoder, and the FLUX latent shift/scale lives in ``comfy.latent_formats.Flux``
(applied by diffusion model sampling). This module calls the autoencoder directly
and applies only the ImageSR scale, exactly once, and never routes latents
through ``latent_formats``.

Precision follows the reference: encoding runs in the VAE dtype without autocast;
decoding runs under CUDA autocast when the VAE dtype is FP16/BF16 (the official
``_autocast`` helper), which keeps group normalisation in FP32 on CUDA.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def decode_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class ImageSRVAE:
    """Encode/decode tiles through a ``first_stage_model`` with ImageSR scaling.

    ``autoencoder`` must expose ``encode(x) -> posterior mode`` and ``decode(z)`` on
    ``[-1, 1]`` pixels, as ``comfy.sd.VAE().first_stage_model`` does for the FLUX VAE.
    """

    def __init__(self, autoencoder: torch.nn.Module, scaling_factor: float, device: torch.device, dtype: torch.dtype):
        self.autoencoder = autoencoder
        self.scaling_factor = float(scaling_factor)
        self.device = device
        self.dtype = dtype

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """FP32 pixels in [0, 1] ``(B, 3, H, W)`` -> FP32 scaled latents on ``pixels.device``."""
        vae_input = (pixels * 2.0 - 1.0).to(device=self.device, dtype=self.dtype)
        latents = self.autoencoder.encode(vae_input)
        if isinstance(latents, tuple):  # (z, reg_log)
            latents = latents[0]
        return latents.float().to(pixels.device) * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """FP32 scaled latents -> FP32 pixels in [0, 1] on ``latents.device``."""
        scaled = (latents / self.scaling_factor).to(device=self.device, dtype=self.dtype)
        with decode_autocast(self.device, self.dtype):
            decoded = self.autoencoder.decode(scaled)
        del scaled
        return ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5).to(latents.device)

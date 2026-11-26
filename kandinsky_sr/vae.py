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


# --------------------------------------------------------------------------- FLUX VAE identity

#: Tensors (ComfyUI/LDM names) whose values identify the FLUX.1 autoencoder.
FINGERPRINT_TENSORS = {
    "encoder": ("encoder.conv_in.weight", "encoder.down.0.block.0.conv1.weight", "encoder.conv_out.weight"),
    "decoder": ("decoder.conv_in.weight", "decoder.up.3.block.0.conv1.weight", "decoder.conv_out.weight"),
}
FINGERPRINT_SAMPLES = 32
#: Relative tolerance: covers BF16/FP16 storage of the FP32 reference weights.
FINGERPRINT_RTOL = 1e-2
_FINGERPRINT_FILE = "flux_vae_fingerprint.json"


def tensor_fingerprint(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    step = max(1, flat.numel() // FINGERPRINT_SAMPLES)
    samples = flat[::step][:FINGERPRINT_SAMPLES]
    return {"shape": list(tensor.shape), "samples": samples.tolist(), "sumsq": float(flat.square().sum())}


def _fingerprint_matches(actual: dict, expected: dict) -> bool:
    if actual["shape"] != expected["shape"] or len(actual["samples"]) != len(expected["samples"]):
        return False
    scale = max(abs(v) for v in expected["samples"]) or 1.0
    for a, e in zip(actual["samples"], expected["samples"]):
        if abs(a - e) > FINGERPRINT_RTOL * max(abs(e), 0.05 * scale):
            return False
    return abs(actual["sumsq"] - expected["sumsq"]) <= FINGERPRINT_RTOL * expected["sumsq"]


def load_reference_fingerprint() -> dict:
    import json
    from pathlib import Path

    return json.loads((Path(__file__).with_name(_FINGERPRINT_FILE)).read_text())


def _expected_ldm_shapes() -> dict[str, tuple[int, ...]]:
    """FLUX VAE parameter shapes under ComfyUI's (LDM) names."""
    import comfy.diffusers_convert

    from .config import VAE_FLUX_CONFIG
    from .loader import expected_vae_shapes

    meta = {k: torch.empty(v, device="meta") for k, v in expected_vae_shapes(VAE_FLUX_CONFIG).items()}
    return {k: tuple(v.shape) for k, v in comfy.diffusers_convert.convert_vae_state_dict(meta).items()}


class IncompatibleVAEError(ValueError):
    """The connected VAE cannot be used with Kandinsky ImageSR."""


def validate_imagesr_vae(vae: object) -> None:
    """Check that ``vae`` (a ``comfy.sd.VAE``) is the FLUX.1 autoencoder this model needs.

    Architecture, latent layout and normalisation are checked structurally; the
    encoder weights must match the FLUX.1 VAE (the latent space the model was
    trained in), so architecturally identical VAEs such as the SD3 VAE are
    rejected. A decoder that differs from FLUX.1 is accepted with a warning (for
    example a fine-tuned decoder) because it does not change the latent space.
    Nothing on the object is modified.
    """
    import logging

    import comfy.ldm.models.autoencoder as autoencoder

    hint = "Connect the FLUX.1 VAE (ae.safetensors) from a Load VAE node."
    model = getattr(vae, "first_stage_model", None)
    if model is None or not hasattr(vae, "patcher"):
        raise IncompatibleVAEError(f"the vae input is not a ComfyUI VAE object. {hint}")
    problems = []
    if type(model) is not autoencoder.AutoencodingEngine:
        problems.append(f"autoencoder type {type(model).__name__}")
    regularizer = getattr(model, "regularization", None)
    if not isinstance(regularizer, autoencoder.DiagonalGaussianRegularizer) or regularizer.sample:
        problems.append("no deterministic diagonal Gaussian posterior")
    if getattr(model, "bn", None) is not None:
        problems.append("batch-normalised latents")
    if getattr(vae, "latent_channels", None) != 16:
        problems.append(f"{getattr(vae, 'latent_channels', None)} latent channels")
    if getattr(vae, "latent_dim", 2) != 2 or getattr(vae, "output_channels", 3) != 3:
        problems.append("not a 2-D RGB image VAE")
    if getattr(vae, "downscale_ratio", None) != 8 or getattr(vae, "upscale_ratio", None) != 8:
        problems.append("spatial compression is not 8x")
    if not problems:
        expected = _expected_ldm_shapes()
        actual = {k: tuple(v.shape) for k, v in model.state_dict(keep_vars=True).items()}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))[:3]
            unexpected = sorted(set(actual) - set(expected))[:3]
            mismatched = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])[:3]
            problems.append(
                f"layout differs from the FLUX VAE (missing {missing}, extra {unexpected}, shape {mismatched})"
            )
    if problems:
        raise IncompatibleVAEError("incompatible VAE: " + "; ".join(problems) + f". {hint}")
    vae_dtype = getattr(vae, "vae_dtype", None)
    if vae_dtype not in (torch.bfloat16, torch.float32):
        raise IncompatibleVAEError(
            f"the VAE runs in {vae_dtype}; Kandinsky ImageSR needs it in bfloat16 or float32 (the FLUX VAE is not "
            "numerically safe in float16). Start ComfyUI with --bf16-vae or --fp32-vae."
        )

    reference = load_reference_fingerprint()
    state = model.state_dict(keep_vars=True)
    for part, names in FINGERPRINT_TENSORS.items():
        matches = all(_fingerprint_matches(tensor_fingerprint(state[n]), reference[n]) for n in names)
        if matches:
            continue
        if part == "encoder":
            raise IncompatibleVAEError(
                "the VAE has the FLUX architecture but its encoder weights are not those of the FLUX.1 VAE "
                f"(for example the SD3 VAE has the same layout and a different latent space). {hint}"
            )
        logging.getLogger(__name__).warning(
            "Kandinsky ImageSR: the connected VAE's decoder differs from the FLUX.1 VAE; results are not qualified"
        )

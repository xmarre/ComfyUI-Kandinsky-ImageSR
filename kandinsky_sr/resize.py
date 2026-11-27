"""Final output scale: resampling of the completed native restoration.

The model restores only at its calibrated native factors (2x and 4x). A different
final size is produced by restoring at a native factor and then resizing the
complete, blended and cropped image to dimensions computed from the *original*
input size:

    target = round_half_up(input * output_scale)

with ``output_scale`` taken as a decimal rounded to six places (so float noise
such as ``1.2000000000000002`` from a UI widget cannot change a dimension). The
resize is antialiased bicubic interpolation in FP32 on NCHW tensors with explicit
integer target dimensions, clamped to [0, 1]. Native restoration, its tile
geometry and its creativity calibration are not affected.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import torch
import torch.nn.functional as F

#: ``output_scale`` value meaning "native restoration size".
NATIVE = 0.0
MIN_OUTPUT_SCALE = 1.0
MAX_OUTPUT_SCALE = 4.0
_QUANTUM = Decimal("0.000001")


def validate_output_scale(output_scale: object, upscale_factor: int) -> Decimal | None:
    """The requested scale as a decimal, or ``None`` for the native output size.

    Raises ``ValueError`` for anything that is not ``0`` or a finite number in
    ``[1, upscale_factor]``. Nothing is clamped or substituted.
    """
    if isinstance(output_scale, bool) or not isinstance(output_scale, (int, float)):
        raise ValueError(f"output_scale must be a number, got {type(output_scale).__name__} {output_scale!r}")
    value = float(output_scale)
    if not math.isfinite(value):
        raise ValueError(f"output_scale must be finite, got {output_scale!r}")
    if value == NATIVE:
        return None
    scale = Decimal(repr(value)).quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    if scale < MIN_OUTPUT_SCALE:
        raise ValueError(
            f"output_scale must be 0 (native size) or at least {MIN_OUTPUT_SCALE}, got {output_scale!r}; "
            "the restoration cannot produce an image smaller than its input"
        )
    if scale > MAX_OUTPUT_SCALE:
        raise ValueError(f"output_scale must not exceed {MAX_OUTPUT_SCALE}, got {output_scale!r}")
    if scale > upscale_factor:
        raise ValueError(
            f"output_scale {output_scale!r} exceeds the native restoration scale {upscale_factor}x; select the "
            "4x upscale_factor for output scales above 2"
        )
    return scale


def scaled_length(length: int, scale: Decimal) -> int:
    """``round_half_up(length * scale)``, at least 1."""
    return max(1, int((Decimal(length) * scale).to_integral_value(rounding=ROUND_HALF_UP)))


def target_size(height: int, width: int, scale: Decimal | None, upscale_factor: int) -> tuple[int, int]:
    """Final ``(height, width)`` for an input of ``height x width``."""
    if scale is None:
        return height * upscale_factor, width * upscale_factor
    return scaled_length(height, scale), scaled_length(width, scale)


def resize_rgb(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """The documented final resize of one BHWC RGB batch: FP32 NCHW bicubic, antialiased, clamped.

    Returns ``(B, *size, 3)`` FP32 on ``image.device``.
    """
    nchw = image.permute(0, 3, 1, 2).to(torch.float32).contiguous()
    resized = F.interpolate(nchw, size=size, mode="bicubic", align_corners=False, antialias=True)
    del nchw
    return resized.clamp_(0.0, 1.0).permute(0, 2, 3, 1)

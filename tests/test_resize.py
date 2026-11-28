"""Final output scale: validation, exact geometry and the documented resize."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest
import torch
import torch.nn.functional as F

from kandinsky_sr.resize import resize_rgb, scaled_length, target_size, validate_output_scale


@pytest.mark.parametrize("value", [0, 0.0, -0.0])
def test_zero_means_native_size(value):
    assert validate_output_scale(value, 2) is None
    assert target_size(37, 51, None, 2) == (74, 102)
    assert target_size(37, 51, None, 4) == (148, 204)


@pytest.mark.parametrize(
    "value,factor,message",
    [
        (-1.0, 2, "at least 1.0"),
        (-0.01, 4, "at least 1.0"),
        (0.5, 2, "at least 1.0"),
        (0.999, 4, "at least 1.0"),
        (math.nan, 2, "finite"),
        (math.inf, 4, "finite"),
        (-math.inf, 4, "finite"),
        (2.01, 2, "select the 4x"),
        (2.5, 2, "select the 4x"),
        (4.01, 4, "must not exceed 4.0"),
        (5, 4, "must not exceed 4.0"),
        (0.9999999, 2, "at least 1.0"),  # would round to 1.000000; rejected before rounding
        (2.0000001, 2, "select the 4x"),
        (4.0000001, 4, "must not exceed 4.0"),
        (1e100, 4, "must not exceed 4.0"),  # never reaches decimal quantisation
        (1.7976931348623157e308, 4, "must not exceed 4.0"),
        (-1e100, 4, "at least 1.0"),
        (True, 2, "must be a number"),
        ("1.5", 2, "must be a number"),
        (None, 2, "must be a number"),
        ([1.5], 2, "must be a number"),
    ],
)
def test_invalid_output_scales_are_rejected_not_clamped(value, factor, message):
    with pytest.raises(ValueError, match=message):
        validate_output_scale(value, factor)


@pytest.mark.parametrize("value,factor", [(1, 2), (1.0, 4), (1.2, 2), (2, 2), (2.0, 4), (2.25, 4), (3.2, 4), (4, 4)])
def test_valid_output_scales(value, factor):
    assert validate_output_scale(value, factor) == Decimal(str(float(value)))


def test_endpoint_tolerance_covers_only_floating_point_noise():
    assert validate_output_scale(1.0 - 1e-12, 2) == Decimal(1)
    assert validate_output_scale(2.0 + 1e-12, 2) == Decimal(2)
    assert validate_output_scale(4.0 + 1e-12, 4) == Decimal(4)
    for value, factor in ((1.0 - 2e-9, 2), (2.0 + 2e-9, 2), (4.0 + 2e-9, 4)):
        with pytest.raises(ValueError):
            validate_output_scale(value, factor)


def test_float_noise_from_widgets_does_not_change_the_scale():
    assert validate_output_scale(1.2000000000000002, 2) == Decimal("1.2")
    assert validate_output_scale(1.2999999999999998, 2) == Decimal("1.3")
    assert validate_output_scale(2.0000000000000004, 2) == Decimal("2")  # not "exceeds 2x"
    # 5 * 1.3 = 6.5 rounds half up to 7 whichever float neighbour of 1.3 arrives
    for noisy in (1.3, 1.2999999999999998, 1.3000000000000003):
        assert scaled_length(5, validate_output_scale(noisy, 2)) == 7


@pytest.mark.parametrize(
    "height,width,scale,expected",
    [
        (1000, 1000, "1.2", (1200, 1200)),
        (1000, 1000, "1.5", (1500, 1500)),
        (1200, 1600, "1.2", (1440, 1920)),
        (1000, 1000, "1", (1000, 1000)),
        (1001, 999, "1.2", (1201, 1199)),  # 1201.2 -> 1201, 1198.8 -> 1199
        (5, 7, "1.5", (8, 11)),  # 7.5 -> 8, 10.5 -> 11: half up, not half to even
        (3, 9, "1.5", (5, 14)),  # 4.5 -> 5, 13.5 -> 14
        (2, 6, "1.75", (4, 11)),  # 3.5 -> 4, 10.5 -> 11
        (1, 1, "1.5", (2, 2)),
        (1, 1, "1.2", (1, 1)),
        (1, 3, "1.01", (1, 3)),
        (17, 33, "2.25", (38, 74)),  # 38.25, 74.25
        (17, 33, "2.5", (43, 83)),  # 42.5 -> 43, 82.5 -> 83
        (17, 33, "3.2", (54, 106)),  # 54.4, 105.6
        (17, 33, "4", (68, 132)),
    ],
)
def test_target_dimensions_round_half_up_from_the_input_size(height, width, scale, expected):
    assert target_size(height, width, Decimal(scale), 4) == expected


def test_aspect_ratio_is_preserved_up_to_rounding():
    for height, width in [(1200, 1600), (333, 777), (1, 1000), (999, 2)]:
        for scale in ("1.2", "1.37", "1.75", "2.5", "3.99"):
            th, tw = target_size(height, width, Decimal(scale), 4)
            assert th >= 1 and tw >= 1
            assert abs(th - height * float(scale)) <= 0.5 and abs(tw - width * float(scale)) <= 0.5


def _image(batch, h, w, seed=0):
    return torch.rand(batch, h, w, 3, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("size", [(29, 36), (24, 30), (7, 5), (1, 1)])
def test_resize_is_antialiased_bicubic_on_nchw_fp32(size):
    image = _image(2, 48, 60)
    out = resize_rgb(image, size)
    assert out.shape == (2, *size, 3) and out.dtype == torch.float32
    expected = F.interpolate(
        image.permute(0, 3, 1, 2).contiguous(), size=size, mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    assert torch.equal(out, expected.permute(0, 2, 3, 1))


def test_resize_overshoot_is_clamped_to_the_image_range():
    image = torch.zeros(1, 40, 40, 3)
    image[:, :, 20:] = 1.0  # a hard edge makes bicubic ring
    raw = F.interpolate(image.permute(0, 3, 1, 2), size=(30, 30), mode="bicubic", align_corners=False, antialias=True)
    assert raw.min() < 0 or raw.max() > 1  # the clamp is not a no-op here
    out = resize_rgb(image, (30, 30))
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_resize_handles_half_precision_input_in_fp32():
    image = _image(1, 32, 32)
    out = resize_rgb(image.to(torch.bfloat16), (20, 20))
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, resize_rgb(image.to(torch.bfloat16).float(), (20, 20)), rtol=0, atol=0)


def test_constant_image_stays_constant():
    image = torch.full((1, 50, 70, 3), 0.375)
    out = resize_rgb(image, (31, 43))
    torch.testing.assert_close(out, torch.full_like(out, 0.375), rtol=0, atol=1e-6)

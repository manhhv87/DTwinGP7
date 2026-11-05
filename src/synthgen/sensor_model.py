"""
sensor_model.py
───────────────
The camera half of the synthetic generator: colour encoding, white balance and
sensor noise, applied to the scene-linear image that Cycles renders.

Colour convention (contract between config/synthgen.yaml, the scene specs and the
renderer): every configured or sampled colour is an sRGB-ENCODED value in [0, 1],
that is a measured 8-bit camera value divided by 255. Blender's Base Color input is
scene-linear, so the renderer passes every colour through srgb_to_linear first.
Writing the sRGB values straight in, as the renderer did until November 2026, made
dark colours several times too bright: the plastic box rendered at (43, 83, 100)
against a measured (8, 38, 57).

The image goes the other way at the end: scene-linear render, times the white
balance and the gain, plus Poisson-Gaussian noise, then the sRGB transfer curve
(Blender's "Standard" view transform, no tone mapping) and 8-bit rounding. The noise
is added in the linear domain because that is where a sensor makes it; the curve
then amplifies it in the shadows, which is why the real hard-condition frames are
so grainy at luma 15.

Pure numpy, plus cv2 for the blur that gives the noise the grain of the real
camera. No package-relative imports: render_blenderproc.py loads this file by path
inside Blender, whose Python has neither pydantic nor the rest of src.synthgen.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Rec. 709 / sRGB primaries: luminance weights of LINEAR RGB.
LUMINANCE_709 = np.array([0.2126, 0.7152, 0.0722])
# Rec. 601 weights on the 8-bit (gamma-encoded) values: the "brightness" of
# src/perception/condition_check.py, so the generator and the cell agree on it.
LUMA_601 = np.array([0.299, 0.587, 0.114])
# Size of the box filter of the noise measure (highpass_noise_std).
NOISE_BOX_PX = 7

# Keys of the sensor section in config/synthgen.yaml, all required.
SENSOR_KEYS = ("gain", "white_balance_rgb", "shot_noise", "read_noise", "grain_sigma_px")


def srgb_to_linear(c: Any) -> np.ndarray:
    """sRGB-encoded value(s) in [0, 1] to linear (IEC 61966-2-1)."""
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: Any) -> np.ndarray:
    """Linear value(s) to sRGB encoding (IEC 61966-2-1); input clipped to [0, 1]."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def kelvin_to_srgb(temp_k: float) -> np.ndarray:
    """Approximate blackbody colour (Tanner Helland fit), sRGB-encoded, 0-1.

    The fit was made on display colours, so its output is sRGB-encoded like every
    other colour here and has to be linearised before it becomes a light colour.
    """
    t = max(1000.0, min(12000.0, float(temp_k))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    b = 255.0 if t >= 66 else (
        0.0 if t <= 19 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    )
    return np.clip(np.array([r, g, b]), 0.0, 255.0) / 255.0


def light_rgb(temp_k: float, white_k: float) -> np.ndarray:
    """Linear colour of a lamp at temp_k, as a camera white-balanced for white_k sees it.

    The camera of the cell balances its white automatically, so a lamp at the
    reference temperature renders neutral: the 5000 K anchor lamp no longer tints the
    grey table to (163, 157, 151) where the camera recorded a neutral (150, 151, 150).
    A lamp at any other temperature keeps its tint relative to that white. The colour
    is scaled to unit luminance, so the lamp's energy alone sets how bright it is.
    """
    rgb = srgb_to_linear(kelvin_to_srgb(temp_k)) / srgb_to_linear(kelvin_to_srgb(white_k))
    return rgb / float(LUMINANCE_709 @ rgb)


def check_sensor(sensor: dict[str, Any]) -> None:
    """Raise ValueError when the sensor section is incomplete or out of range."""
    missing = [k for k in SENSOR_KEYS if k not in sensor]
    if missing:
        raise ValueError(f"sensor section misses {missing}")
    if float(sensor["gain"]) <= 0.0:
        raise ValueError("sensor.gain must be positive")
    wb = np.asarray(sensor["white_balance_rgb"], dtype=float)
    if wb.shape != (3,) or np.any(wb <= 0.0):
        raise ValueError("sensor.white_balance_rgb must be three positive gains")
    for key in ("shot_noise", "read_noise", "grain_sigma_px"):
        if float(sensor[key]) < 0.0:
            raise ValueError(f"sensor.{key} cannot be negative")


def unit_grain(shape: tuple[int, ...], sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance noise field with the grain of the real camera.

    White Gaussian noise blurred by a Gaussian of sigma_px in each channel, then
    rescaled to unit variance. The D455's processing correlates neighbouring pixels:
    the noise residual of its frames correlates 0.70 / 0.33 / 0.05 at 1 / 2 / 3 px,
    which white noise cannot reproduce (it correlates 0 at every lag).
    """
    n = rng.standard_normal(shape).astype(np.float32)
    if sigma_px <= 0.0:
        return n
    import cv2  # lazy import: only needed when the grain is on

    ksize = 2 * int(np.ceil(3.0 * sigma_px)) + 1
    k1 = cv2.getGaussianKernel(ksize, sigma_px).ravel()
    norm = 1.0 / float(np.sum(k1 ** 2))     # 1 / sum of the 2-D kernel's squares
    if n.ndim == 2:
        return cv2.GaussianBlur(n, (ksize, ksize), sigma_px,
                                borderType=cv2.BORDER_REFLECT) * norm
    out = np.empty_like(n)
    for c in range(n.shape[2]):
        out[..., c] = cv2.GaussianBlur(n[..., c], (ksize, ksize), sigma_px,
                                       borderType=cv2.BORDER_REFLECT) * norm
    return out


def apply_sensor(linear_rgb: np.ndarray, sensor: dict[str, Any],
                 rng: np.random.Generator | None, exposure_gain: float = 1.0) -> np.ndarray:
    """Scene-linear render (H, W, 3) to the 8-bit sRGB frame the camera would give.

    y = gain * white_balance * x, then noise of variance
        gain * shot_noise * y + gain**2 * read_noise
    (Poisson photon noise, whose variance grows with the signal, plus read noise,
    both amplified by the analogue gain), with the grain of unit_grain, then the sRGB
    curve and rounding to 8 bits. gain = sensor["gain"] * exposure_gain:
    sensor["gain"] is the gain at which the profile was fitted (the auto-exposure
    state of the condition check), exposure_gain the scene's session exposure that
    the sampler draws (spec["exposure"]["gain"]). In the dim light of the hard
    condition the camera's automatic exposure does not hold the image level, and the
    noise rises with the level as a gain change predicts: 2.03 at luma 21.5 and 2.54
    at 30.3 on two condition checks, a ratio of 1.25, against 1.26 for a gain change
    and 0.93 for a change of light at fixed gain.
    rng None leaves the noise out: the noise-free expectation, used to fit the profile.
    """
    check_sensor(sensor)
    if not float(exposure_gain) > 0.0:
        raise ValueError(f"exposure_gain must be positive, got {exposure_gain}")
    gain = float(sensor["gain"]) * float(exposure_gain)
    y = np.asarray(linear_rgb, dtype=np.float32) * (
        gain * np.asarray(sensor["white_balance_rgb"], dtype=np.float32))
    if rng is not None:
        var = (gain * float(sensor["shot_noise"]) * np.clip(y, 0.0, None)
               + gain ** 2 * float(sensor["read_noise"]))
        y = y + np.sqrt(var) * unit_grain(y.shape, float(sensor["grain_sigma_px"]), rng)
    srgb = linear_to_srgb(y)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def luma601(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of RGB values (any range), as condition_check.py computes it."""
    return np.asarray(rgb, dtype=float) @ LUMA_601


def highpass_noise_std(gray: np.ndarray) -> float:
    """Noise measure used for real and synthetic frames alike.

    Standard deviation of the residual after subtracting a 7x7 box mean, over the
    interior of the patch: smooth shading drops out, pixel-scale noise stays. Real
    hard-condition frames give 1.6 to 1.9 on the green sheet at luma 15.
    """
    import cv2  # lazy import

    g = np.asarray(gray, dtype=np.float32)
    hp = g - cv2.blur(g, (NOISE_BOX_PX, NOISE_BOX_PX))
    m = NOISE_BOX_PX // 2 + 1
    return float(hp[m:-m, m:-m].std())

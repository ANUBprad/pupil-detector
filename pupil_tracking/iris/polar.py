"""Geometry-aware polar iris registration (additive, experimental).

Reference-derived capability, integrated additively: the rubber-sheet unwrap
consumes the existing ellipse-aware ``IrisROI`` geometry via
:class:`IrisNormalizer` (NOT the reference implementation's circular
pupil/limbus model), reflection masking reuses the existing
:class:`IrisMasking`, and angular registration is performed by phase /
circular correlation on a gradient channel of an illumination-flattened polar
image.

Scope and safety
----------------
* Disabled by default (``PolarRegistrationConfig.enabled = False``). Nothing
  here runs in the normal detection path and no rotation value is emitted to
  surgical control, planning, centration, or treatment logic.
* The reference's final fusion (median of KAZE/phase/circular + spread labels)
  is NOT used. Estimates are kept independent and gated by the existing
  evidence culture: geometry validity, usable iris area, texture/SNR, angular
  coverage, peak ambiguity, and cross-method consistency. Anything short
  returns an honest ``valid=False`` refusal.
* KAZE feature matching is intentionally not ported: Phase 16A showed
  fine-grained descriptor correspondence is rotation-fragile, and the existing
  correspondence engine already provides feature-level matching.

Rotation convention (matches ``correspondence.py``): an applied positive
(clockwise) image rotation is reported as a positive rotation estimate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.iris.masking import IrisMasking
from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import IrisFeatureSet, IrisROI


# --------------------------------------------------------------------------- #
# Configuration / result contracts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PolarRegistrationConfig:
    """Tunables for the polar registration component (experimental)."""

    enabled: bool = False

    angular_resolution: int = 1440
    radial_resolution: int = 128

    # illumination flattening of the polar image
    illum_sigma_deg: float = 15.0       # background Gauss sigma (columns ~ deg)

    # rotation search
    max_rotation_deg: float = 15.0
    search_step_deg: float = 0.25       # 1 column at 1440 columns / 360 deg

    # evidence gates
    min_usable_fraction: float = 0.15   # fraction of annulus usable (per frame)
    min_column_usable_frac: float = 0.30  # per angular column usable fraction
    min_angular_coverage_frac: float = 0.50  # fraction of columns above the above
    min_texture_gradient: float = 5.0   # mean Sobel magnitude over usable polar
    min_phase_response: float = 0.010   # phase correlation response floor
    min_circular_score: float = 0.05    # circular correlation score floor
    peak_exclusion_deg: float = 1.0     # second-best must be >= this away
    min_peak_margin: float = 0.02       # best - second-best score margin
    max_method_spread_deg: float = 0.75  # phase vs circular disagreement limit

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "angular_resolution": self.angular_resolution,
            "radial_resolution": self.radial_resolution,
            "illum_sigma_deg": self.illum_sigma_deg,
            "max_rotation_deg": self.max_rotation_deg,
            "search_step_deg": self.search_step_deg,
            "min_usable_fraction": self.min_usable_fraction,
            "min_column_usable_frac": self.min_column_usable_frac,
            "min_angular_coverage_frac": self.min_angular_coverage_frac,
            "min_texture_gradient": self.min_texture_gradient,
            "min_phase_response": self.min_phase_response,
            "min_circular_score": self.min_circular_score,
            "peak_exclusion_deg": self.peak_exclusion_deg,
            "min_peak_margin": self.min_peak_margin,
            "max_method_spread_deg": self.max_method_spread_deg,
        }


class PolarFailureKind(str):
    """Why a polar registration produced no valid rotation."""

    OK = "OK"
    DISABLED = "DISABLED"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    LOW_USABLE_AREA = "LOW_USABLE_AREA"
    LOW_ANGULAR_COVERAGE = "LOW_ANGULAR_COVERAGE"
    LOW_TEXTURE = "LOW_TEXTURE"
    NONE = "NONE"


@dataclass
class PolarRegistrationResult:
    """Independent angular-registration estimate for one image pair."""

    valid: bool = False
    failure: str = PolarFailureKind.NONE
    failure_reason: str = ""

    rotation_deg: float = 0.0
    phase_rotation_deg: Optional[float] = None
    phase_response: Optional[float] = None
    circular_rotation_deg: Optional[float] = None
    circular_score: Optional[float] = None
    second_best_score: Optional[float] = None
    peak_margin: Optional[float] = None
    method_spread_deg: Optional[float] = None

    usable_fraction_a: float = 0.0
    usable_fraction_b: float = 0.0
    angular_coverage: float = 0.0
    texture_gradient_a: float = 0.0
    texture_gradient_b: float = 0.0

    polar_shape: Tuple[int, int] = (0, 0)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "valid": bool(self.valid),
            "failure": self.failure,
            "failure_reason": self.failure_reason,
            "rotation_deg": float(self.rotation_deg),
            "phase_rotation_deg": self.phase_rotation_deg,
            "phase_response": self.phase_response,
            "circular_rotation_deg": self.circular_rotation_deg,
            "circular_score": self.circular_score,
            "second_best_score": self.second_best_score,
            "peak_margin": self.peak_margin,
            "method_spread_deg": self.method_spread_deg,
            "usable_fraction_a": self.usable_fraction_a,
            "usable_fraction_b": self.usable_fraction_b,
            "angular_coverage": self.angular_coverage,
            "texture_gradient_a": self.texture_gradient_a,
            "texture_gradient_b": self.texture_gradient_b,
            "polar_shape": list(self.polar_shape),
            "processing_time_ms": self.processing_time_ms,
        }
        return d


# --------------------------------------------------------------------------- #
# Polar unwrap (geometry-aware, ellipse ROI)
# --------------------------------------------------------------------------- #

def unwrap_iris(
    gray: np.ndarray,
    roi: IrisROI,
    angular_resolution: int = 1440,
    radial_resolution: int = 128,
    usable: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """Unwrap the iris annulus into (radial, angular) polar space.

    Every angular column samples along the *ellipse-aware* radial bounds from
    :meth:`IrisNormalizer.radial_bounds`, so pupil-offset and non-circular
    geometry is honoured instead of the concentric-circle simplification.

    Returns ``(polar, polar_usable, coverage_stats)``. ``polar``/``polar_usable``
    are None when the geometry is unusable (invalid ROI, inner radius >= outer
    radius, or the image is too small). ``coverage_stats`` holds
    ``mean_column_valid``, ``angular_coverage`` and ``largest_angular_gap_deg``.
    """
    empty = {"mean_column_valid": 0.0, "angular_coverage": 0.0,
             "largest_angular_gap_deg": 360.0}
    if gray is None or gray.size == 0 or not roi.valid:
        return None, None, empty
    h, w = gray.shape[:2]
    norm = IrisNormalizer()
    ares = int(angular_resolution)
    rres = int(radial_resolution)

    angles = np.linspace(0.0, 360.0, ares, endpoint=False)
    inner = np.empty(ares, dtype=np.float64)
    outer = np.empty(ares, dtype=np.float64)
    for i, ang in enumerate(angles):
        ri, ro = norm.radial_bounds(roi, float(ang))
        inner[i], outer[i] = ri, ro
    if np.any(outer <= inner) or np.mean(outer - inner) <= 1.0:
        return None, None, empty

    radii = inner[:, None] + np.linspace(0.0, 1.0, rres)[None, :] * (
        outer - inner
    )[:, None]
    theta = np.deg2rad(angles)
    map_x = (roi.center_x + radii * np.cos(theta[:, None])).astype(np.float32)
    map_y = (roi.center_y + radii * np.sin(theta[:, None])).astype(np.float32)
    # remap needs (columns, rows) maps: each output pixel (r, a) at map_x[a, r].
    mx = np.ascontiguousarray(map_x.T)
    my = np.ascontiguousarray(map_y.T)
    src = gray.astype(np.float32, copy=False) if gray.dtype != np.float32 else gray
    polar = cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)

    polar_usable = None
    if usable is not None and usable.shape == gray.shape[:2]:
        usable_f = usable.astype(np.float32)
        pol_mask = cv2.remap(usable_f, mx, my,
                             interpolation=cv2.INTER_NEAREST,
                             borderValue=0.0)
        polar_usable = pol_mask > 0.5
        col_count = polar_usable.sum(axis=0)
        col_valid = col_count / float(rres)
        covered = col_valid >= 0.30
        return (polar, polar_usable,
                {"mean_column_valid": float(col_valid.mean()),
                 "angular_coverage": float(covered.mean()) if ares else 0.0,
                 "largest_angular_gap_deg": _largest_gap(covered)})

    return polar, polar_usable, empty


def _largest_gap(covered: np.ndarray) -> float:
    if covered.size == 0 or not covered.any():
        return 360.0
    idx = np.flatnonzero(covered)
    edges = np.diff(idx)
    max_inner = int(edges.max()) if edges.size else 0
    wrap = int((idx[0] + covered.size - idx[-1]) % covered.size)
    return float(max(max_inner, wrap) * 360.0 / covered.size)


def normalize_polar(
    polar: np.ndarray,
    illum_sigma_deg: float = 15.0,
) -> np.ndarray:
    """Flatten slow (along-column) illumination variation, range-normalise."""
    if polar is None or polar.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    axis_order = polar.ndim
    background = cv2.GaussianBlur(
        polar, (0, 0), sigmaX=float(illum_sigma_deg)
    ) if axis_order == 2 else polar
    diff = polar - background
    lo, hi = float(diff.min()), float(diff.max())
    span = hi - lo
    if span <= 1e-6:
        return np.zeros(polar.shape, dtype=np.uint8)
    out = (diff - lo) / span * 255.0
    return out.astype(np.uint8)


def gradient_magnitude(polar: np.ndarray) -> np.ndarray:
    """Sobel magnitude channel used for the correlation methods."""
    if polar is None or polar.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gx = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(polar, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Angular registration
# --------------------------------------------------------------------------- #

def shift_to_rotation_deg(shift_columns: float, width: int) -> float:
    """Convert a polar column shift to degrees, wrapped to (-180, 180]."""
    deg = -float(shift_columns) / float(width) * 360.0
    return ((deg + 180.0) % 360.0) - 180.0


def phase_rotation(
    polar_a: np.ndarray,
    polar_b: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Phase-correlation rotation estimate ``(deg, response)``.

    Returns ``(None, None)`` on degenerate input (empty/zero-energy after
    DC removal). Response is the ``cv2.phaseCorrelate`` response magnitude.
    """
    if polar_a is None or polar_b is None:
        return None, None
    if polar_a.shape != polar_b.shape or polar_a.size == 0:
        return None, None
    h, w = polar_a.shape[:2]
    a = polar_a.astype(np.float32)
    b = polar_b.astype(np.float32)
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    norm_a = float(np.linalg.norm(a_mean))
    norm_b = float(np.linalg.norm(b_mean))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None, None
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a_mean, b_mean, window)
    return shift_to_rotation_deg(float(shift[0]), w), float(response)


def circular_rotation_scan(
    polar_a: np.ndarray,
    polar_b: np.ndarray,
    max_angle_deg: float = 15.0,
    step_deg: float = 0.25,
    exclusion_deg: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Exhaustive angular scan over ``[-max_angle, +max_angle]``.

    Returns ``(best_deg, best_score, second_best_score)`` where
    ``second_best_score`` is the best score at least ``exclusion_deg`` away
    from the winning shift (used for ambiguity / tie detection). None when the
    input is degenerate or the scan has no valid position.
    """
    if polar_a is None or polar_b is None:
        return None, None, None
    if polar_a.shape != polar_b.shape or polar_a.size == 0:
        return None, None, None
    w = polar_a.shape[1]
    a = polar_a.astype(np.float32)
    a = a - np.mean(a)
    norm_a = np.linalg.norm(a)
    if norm_a == 0:
        return None, None, None

    angles = np.arange(-max_angle_deg, max_angle_deg + step_deg, step_deg)
    scores = np.empty(len(angles), dtype=float)
    ok = np.zeros(len(angles), dtype=bool)
    for i, ang in enumerate(angles):
        shift_cols = int(round(ang / 360.0 * w))
        shifted = np.roll(polar_b, shift_cols, axis=1).astype(np.float32)
        shifted -= np.mean(shifted)
        nb = np.linalg.norm(shifted)
        if nb == 0:
            continue
        scores[i] = float(np.sum(a * shifted) / (norm_a * nb))
        ok[i] = True
    if not np.any(ok):
        return None, None, None
    scores[~ok] = -np.inf

    best_i = int(np.argmax(scores))
    best_deg = float(angles[best_i])
    best_score = float(scores[best_i])

    exclusion_cols = max(int(round(exclusion_deg / 360.0 * w)), 1)
    mask = np.arange(len(angles))[:, None]
    rows = np.arange(len(angles))
    far = abs(rows - best_i) > exclusion_cols
    second = float(np.max(scores[far])) if np.any(far) else None
    return best_deg, best_score, second


# --------------------------------------------------------------------------- #
# Gated orchestration
# --------------------------------------------------------------------------- #

def estimate_polar_registration(
    polar_a: np.ndarray,
    polar_b: np.ndarray,
    polar_usable_a: Optional[np.ndarray],
    polar_usable_b: Optional[np.ndarray],
    config: Optional[PolarRegistrationConfig] = None,
) -> PolarRegistrationResult:
    """Run both correlation methods and apply the evidence gates."""
    cfg = config or PolarRegistrationConfig()
    res = PolarRegistrationResult()
    if polar_a is None or polar_b is None or polar_a.size == 0:
        res.failure = PolarFailureKind.NONE
        res.failure_reason = "degenerate polar input"
        return res
    res.polar_shape = (polar_a.shape[0], polar_a.shape[1])

    # Gates (texture/coverage) must reflect ONLY the usable iris, so the
    # masked channels are used for the statistics. The correlation methods
    # themselves run on the FULL normalized polar (as in the reference
    # rubber-sheet correlation); masking out islands in the polar frame
    # introduces zero-padding edges that bias the correlation peak.
    if polar_usable_a is not None:
        masked_a = np.where(polar_usable_a, polar_a, 0.0)
    else:
        masked_a = polar_a
    if polar_usable_b is not None:
        masked_b = np.where(polar_usable_b, polar_b, 0.0)
    else:
        masked_b = polar_b

    gm_a = gradient_magnitude(normalize_polar(masked_a, cfg.illum_sigma_deg))
    gm_b = gradient_magnitude(normalize_polar(masked_b, cfg.illum_sigma_deg))
    res.texture_gradient_a = float(np.mean(gm_a))
    res.texture_gradient_b = float(np.mean(gm_b))
    if (res.texture_gradient_a < cfg.min_texture_gradient
            or res.texture_gradient_b < cfg.min_texture_gradient):
        res.failure = PolarFailureKind.LOW_TEXTURE
        res.failure_reason = (
            f"texture gradient {res.texture_gradient_a:.2f}/{res.texture_gradient_b:.2f}"
            f" below {cfg.min_texture_gradient}"
        )
        return res

    # angular coverage gate (usable polar mask)
    if polar_usable_a is not None and polar_usable_b is not None:
        cov_a = float(np.mean(
            polar_usable_a.sum(axis=0) / polar_usable_a.shape[0] >= 0.30))
        cov_b = float(np.mean(
            polar_usable_b.sum(axis=0) / polar_usable_b.shape[0] >= 0.30))
        res.angular_coverage = min(cov_a, cov_b)
        if res.angular_coverage < cfg.min_angular_coverage_frac:
            res.failure = PolarFailureKind.LOW_ANGULAR_COVERAGE
            res.failure_reason = (
                f"angular coverage {res.angular_coverage:.3f} below "
                f"{cfg.min_angular_coverage_frac}"
            )
            return res

    # correlation channels (full polar)
    ga = gradient_magnitude(normalize_polar(polar_a, cfg.illum_sigma_deg))
    gb = gradient_magnitude(normalize_polar(polar_b, cfg.illum_sigma_deg))
    phase_deg, phase_resp = phase_rotation(ga, gb)
    circ_deg, circ_score, second = circular_rotation_scan(
        ga, gb, max_angle_deg=cfg.max_rotation_deg, step_deg=cfg.search_step_deg,
        exclusion_deg=cfg.peak_exclusion_deg,
    )
    res.phase_rotation_deg = phase_deg
    res.phase_response = phase_resp
    res.circular_rotation_deg = circ_deg
    res.circular_score = circ_score
    res.second_best_score = second
    if phase_deg is not None:
        res.peak_margin = (None if circ_score is None or second is None
                           else float(circ_score - second))

    if phase_deg is None or phase_resp is None or circ_deg is None or circ_score is None:
        res.failure = PolarFailureKind.NONE
        res.failure_reason = "correlation produced no estimate"
        return res

    # correlation strength gates
    if phase_resp < cfg.min_phase_response:
        res.failure = PolarFailureKind.LOW_TEXTURE
        res.failure_reason = f"phase response {phase_resp:.4f} below {cfg.min_phase_response}"
        return res
    if circ_score < cfg.min_circular_score:
        res.failure = PolarFailureKind.LOW_TEXTURE
        res.failure_reason = f"circular score {circ_score:.4f} below {cfg.min_circular_score}"
        return res
    if second is not None and res.peak_margin is not None and res.peak_margin < cfg.min_peak_margin:
        res.failure = PolarFailureKind.LOW_ANGULAR_COVERAGE
        res.failure_reason = (
            f"ambiguous peak (margin {res.peak_margin:.4f} < {cfg.min_peak_margin})"
        )
        return res

    # cross-method consistency gate
    spread = abs(((phase_deg - circ_deg) + 180.0) % 360.0 - 180.0)
    res.method_spread_deg = float(spread)
    if spread > cfg.max_method_spread_deg:
        res.failure = PolarFailureKind.NONE
        res.failure_reason = (
            f"methods disagree ({phase_deg:.2f} vs {circ_deg:.2f}, "
            f"spread {spread:.2f} deg > {cfg.max_method_spread_deg})"
        )
        return res

    res.rotation_deg = float((phase_deg + circ_deg) / 2.0)
    res.valid = True
    res.failure = PolarFailureKind.OK
    return res


def estimate_iris_rotation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_set_a: IrisFeatureSet,
    feature_set_b: IrisFeatureSet,
    config: Optional[PolarRegistrationConfig] = None,
    masking_a: Optional[IrisMasking] = None,
    masking_b: Optional[IrisMasking] = None,
) -> PolarRegistrationResult:
    """End-to-end polar angular registration for two images + feature sets.

    Consumes the existing pupil/limbus-derived ROIs and the existing
    :class:`IrisMasking`; produces an independent, gated rotation estimate that
    never feeds surgical planning (it is per-call experimental output).
    """
    t0 = time.perf_counter()
    cfg = config or PolarRegistrationConfig()
    res = PolarRegistrationResult()
    res.processing_time_ms = 0.0

    def _finish(failure: str, reason: str) -> PolarRegistrationResult:
        res.failure = failure
        res.failure_reason = reason
        res.valid = False
        res.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        return res

    roi_a, roi_b = feature_set_a.roi, feature_set_b.roi
    if not roi_a.valid or not roi_b.valid:
        return _finish(PolarFailureKind.INVALID_GEOMETRY, "invalid ROI")

    ma = (masking_a or IrisMasking()).build(image_a, roi_a)
    mb = (masking_b or IrisMasking()).build(image_b, roi_b)

    def _grab_polar(image, roi, usable):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 \
            else image.astype(np.float32, copy=False)
        polar, pu, stats = unwrap_iris(
            gray, roi, cfg.angular_resolution, cfg.radial_resolution, usable)
        return polar, pu, stats, usable

    polar_a, pu_a, stats_a, usable_a = _grab_polar(image_a, roi_a, ma)
    polar_b, pu_b, stats_b, usable_b = _grab_polar(image_b, roi_b, mb)
    if polar_a is None or polar_b is None:
        return _finish(PolarFailureKind.INVALID_GEOMETRY, "unwrap failed")

    annulus_a = np.pi * (roi_a.limbus_radius_px ** 2 - roi_a.pupil_radius_px ** 2)
    annulus_b = np.pi * (roi_b.limbus_radius_px ** 2 - roi_b.pupil_radius_px ** 2)
    res.usable_fraction_a = (float(np.count_nonzero(usable_a)) / annulus_a
                             if annulus_a > 0 else 0.0)
    res.usable_fraction_b = (float(np.count_nonzero(usable_b)) / annulus_b
                             if annulus_b > 0 else 0.0)
    if (res.usable_fraction_a < cfg.min_usable_fraction
            or res.usable_fraction_b < cfg.min_usable_fraction):
        return _finish(
            PolarFailureKind.LOW_USABLE_AREA,
            f"usable {res.usable_fraction_a:.3f}/{res.usable_fraction_b:.3f} "
            f"below {cfg.min_usable_fraction}")

    res = estimate_polar_registration(polar_a, polar_b, pu_a, pu_b, cfg)
    if annulus_a > 0:
        res.usable_fraction_a = float(np.count_nonzero(usable_a) / annulus_a)
    if annulus_b > 0:
        res.usable_fraction_b = float(np.count_nonzero(usable_b) / annulus_b)
    res.angular_coverage = max(res.angular_coverage, min(
        stats_a["angular_coverage"], stats_b["angular_coverage"]))
    res.processing_time_ms = (time.perf_counter() - t0) * 1000.0
    return res


def detect_iris_rotation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_set_a: IrisFeatureSet,
    feature_set_b: IrisFeatureSet,
    config: Optional[PolarRegistrationConfig] = None,
) -> PolarRegistrationResult:
    """Alias for :func:`estimate_iris_rotation` honoring ``config.enabled``."""
    cfg = config or PolarRegistrationConfig()
    if not cfg.enabled:
        return PolarRegistrationResult(
            valid=False, failure=PolarFailureKind.DISABLED,
            failure_reason="polar registration disabled by default")
    return estimate_iris_rotation(image_a, image_b, feature_set_a,
                                  feature_set_b, cfg)


# --------------------------------------------------------------------------- #
# Deterministic self-check (python -m pupil_tracking.iris.polar)
# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """Sig/behavior check. Fails loudly if the sign convention drifts."""
    rng = np.random.default_rng(3)
    w, r = 1440, 128
    col = np.linspace(0, 1, w)[None, :]
    rr = np.linspace(0, 1, r)[:, None]
    base = (128 + 60 * np.sin(2 * np.pi * 6 * rr)).astype(float)
    texture = (
        base + 80 * np.sin(2 * np.pi * 7 * col)
        + rng.standard_normal((r, w)) * 40.0
    ).clip(0, 255).astype(np.uint8)

    for cols, expect in ((21, -5.25), (-21, 5.25), (0, 0.0)):
        res = estimate_polar_registration(
            texture, np.roll(texture, cols, axis=1), None, None,
            PolarRegistrationConfig())
        ok = res.valid and abs(res.rotation_deg - expect) < 0.3
        print(f"roll {cols:+3d} -> {res.rotation_deg:+.2f} deg (expect {expect:+.2f}): "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(f"self-check failed for roll {cols}")

    flat = np.full((r, w), 60, np.uint8)
    res = estimate_polar_registration(flat, flat, None, None,
                                      PolarRegistrationConfig())
    print(f"flat -> {res.failure}: {'OK' if not res.valid else 'FAIL'}")
    if res.valid:
        raise AssertionError("flat texture must refuse")
    print("polar self-check OK")


if __name__ == "__main__":
    _self_check()
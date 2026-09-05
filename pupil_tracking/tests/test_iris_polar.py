"""Focused tests for the additive polar iris registration module.

Covers the mandated experimental surface: geometry-aware ellipse unwrap,
illumination flattening, gradient channel, phase / circular correlation with
the shared rotation sign convention, and the honest-refusal gates (invalid
geometry, insufficient usable area, low coverage, low texture, ambiguous
peaks, cross-method inconsistency). All inputs are deterministic synthetics
with known angular shifts; no clinical data is used.
"""

import dataclasses
import numpy as np
import pytest

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.detect import detect_iris_features
from pupil_tracking.iris.masking import IrisMasking
from pupil_tracking.iris.paired import PairConfig, make_synthetic_pair
from pupil_tracking.iris.polar import (
    PolarFailureKind,
    PolarRegistrationConfig,
    PolarRegistrationResult,
    circular_rotation_scan,
    detect_iris_rotation,
    estimate_iris_rotation,
    estimate_polar_registration,
    gradient_magnitude,
    normalize_polar,
    phase_rotation,
    shift_to_rotation_deg,
    unwrap_iris,
)
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.utils.types import EllipseParams


W = 1440


# ── fixtures ───────────────────────────────────────────────────────────

def polar_texture(kind: str = "unique") -> np.ndarray:
    """Deterministic (R, W) polar texture with strong unique angular content."""
    rng = np.random.default_rng(3)
    rr = np.linspace(0, 1, 128)[:, None]
    col = np.linspace(0, 1, W)[None, :]
    base = (128 + 60 * np.sin(2 * np.pi * 6 * rr)).astype(float)
    if kind == "unique":
        noise = rng.standard_normal((128, W)) * 40.0
        ang_band = 80 * np.sin(2 * np.pi * 7 * col)
        return (base + ang_band + noise).clip(0, 255).astype(np.uint8)
    if kind == "weak":
        # same periodicity as "unique" but overwhelmed by it: shallow,
        # nearly-ambiguous correlations.
        return (base + rng.standard_normal((128, W)) / 3.0).clip(
            0, 255
        ).astype(np.uint8)
    if kind == "periodic":
        harm = np.zeros((1, W))
        for h in (1, 2, 3, 4, 5):
            harm = harm + (160.0 / h) * np.sin(2 * np.pi * h * 2 * col)
        return np.repeat((128 + harm).clip(0, 255).astype(np.uint8), 128, axis=0)
    if kind == "flat":
        return np.full((128, W), 60, np.uint8)
    raise ValueError(kind)


def angular_iris_bgr(size=320, seed=7, tex=14.0) -> np.ndarray:
    """Synthetic iris with genuine angular structure (spokes + angle ripple)."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0
    dx, dy = xx - c, yy - c
    rad = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)
    spokes = np.where(
        ((np.rad2deg(ang) % 60.0) < 9.0) & (rad > 45) & (rad < 135), 30.0, 0.0
    )
    iris = (70 + 100 * np.sin(rad * 0.35) + 25 * np.sin(3 * ang) + spokes).clip(
        0, 255
    ).astype(np.uint8)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-int(tex), int(tex), iris.shape).astype(np.int16)
    gray = np.clip(iris.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def iris_geometry(size=320):
    c = size // 2
    return (
        EllipseParams(center_x=c, center_y=c, semi_major=55.0,
                      semi_minor=55.0, angle_deg=0.0),
        EllipseParams(center_x=c, center_y=c, semi_major=130.0,
                      semi_minor=130.0, angle_deg=0.0),
    )


@pytest.fixture(scope="module")
def eye_src():
    return angular_iris_bgr()


@pytest.fixture(scope="module")
def eye_geo():
    return iris_geometry()


@pytest.fixture(scope="module")
def fs_a(eye_src, eye_geo):
    pe, le = eye_geo
    res = detect_iris_features(eye_src, pe, le, config=IrisConfig())
    assert res.feature_set.roi.valid
    return res.feature_set


@pytest.fixture(scope="module")
def no_lid():
    return IrisMasking(eyelid_method="none")


@pytest.fixture(scope="module")
def rot_cfg():
    return PolarRegistrationConfig(
        enabled=True, angular_resolution=720, radial_resolution=96
    )


# ── unwrap ─────────────────────────────────────────────────────────────

def test_unwrap_polar_dimensions(fs_a):
    gray = np.zeros((320, 320), dtype=np.float32)
    polar, pu, stats = unwrap_iris(
        gray, fs_a.roi, angular_resolution=W, radial_resolution=96)
    assert polar is not None
    assert polar.shape == (96, W)
    assert pu is None
    assert stats["angular_coverage"] == 0.0  # no usable mask provided


def test_unwrap_invalid_geometry_returns_none():
    roi = IrisROI(valid=False)
    polar, pu, stats = unwrap_iris(np.zeros((64, 64), np.float32), roi)
    assert polar is None and pu is None
    assert stats["angular_coverage"] == 0.0


def test_unwrap_circular_roi_all_columns_uniform():
    size = 200
    yy, xx = np.mgrid[0:size, 0:size]
    gray = np.sqrt((xx - 100.0) ** 2 + (yy - 100.0) ** 2).astype(np.float32)
    roi = IrisROI(
        valid=True, center_x=100.0, center_y=100.0,
        pupil_semi_major=30.0, pupil_semi_minor=30.0, pupil_angle_deg=0.0,
        limbus_semi_major=80.0, limbus_semi_minor=80.0, limbus_angle_deg=0.0,
        pupil_radius_px=30.0, limbus_radius_px=80.0,
        inner_inset_frac=0.0, outer_inset_frac=0.0,
    )
    polar, _, _ = unwrap_iris(gray, roi, angular_resolution=360, radial_resolution=64)
    assert polar is not None
    # concentric rings -> identical radial profile in every column (bilinear tolerance)
    for c in (0, 90, 180, 270, 359):
        np.testing.assert_allclose(polar[:, c], polar[:, 0], atol=0.05)


def test_unwrap_ellipse_aware_nonuniform():
    size = 200
    yy, xx = np.mgrid[0:size, 0:size]
    gray = np.sqrt((xx - 100.0) ** 2 + (yy - 100.0) ** 2).astype(np.float32)
    # elliptical pupil (wider on x-axis) inside a circular limbus
    roi = IrisROI(
        valid=True, center_x=100.0, center_y=100.0,
        pupil_semi_major=50.0, pupil_semi_minor=30.0, pupil_angle_deg=0.0,
        limbus_semi_major=80.0, limbus_semi_minor=80.0, limbus_angle_deg=0.0,
        pupil_radius_px=40.0, limbus_radius_px=80.0,
        inner_inset_frac=0.0, outer_inset_frac=0.0,
    )
    polar, _, _ = unwrap_iris(gray, roi, angular_resolution=360, radial_resolution=64)
    assert polar is not None
    # wider pupil on x-axis -> narrower radial band -> different profile than y-axis
    col_0 = polar[:, 0]
    col_90 = polar[:, 90]
    assert float(np.mean(np.abs(col_0 - col_90))) > 0.5


# ── wrap / sign conventions ────────────────────────────────────────────

def test_shift_to_rotation_deg_sign_and_wrap():
    assert shift_to_rotation_deg(1.0, W) == pytest.approx(-0.25)
    assert shift_to_rotation_deg(-1.0, W) == pytest.approx(0.25)
    val = shift_to_rotation_deg(-1600.0, W)  # -1600 cols -> 400 deg -> wraps to 40
    assert val == pytest.approx(40.0)
    assert -180.0 < val <= 180.0


def test_phase_rotation_recovers_known_shift():
    pa = polar_texture("unique")
    deg, resp = phase_rotation(pa, np.roll(pa, 21, axis=1))
    assert deg is not None and resp is not None
    assert deg == pytest.approx(-21.0 * 360.0 / W, abs=0.15)
    assert resp > 0.9


def test_circular_rotation_scan_sign_and_zero():
    pa = polar_texture("unique")
    deg, score, second = circular_rotation_scan(
        pa, np.roll(pa, 0, axis=1), max_angle_deg=15.0)
    assert deg == pytest.approx(0.0, abs=1e-9)
    assert score is not None and score > 0.95
    assert second is not None


# ── registration estimates ─────────────────────────────────────────────

@pytest.mark.parametrize("cols", [21, -21, 0, 60])
def test_registration_recovers_known_rolls(cols):
    cfg = PolarRegistrationConfig(max_rotation_deg=15.0)
    pa = polar_texture("unique")
    expect = -cols * 360.0 / W
    res = estimate_polar_registration(pa, np.roll(pa, cols, axis=1), None, None, cfg)
    assert res.valid, res.failure_reason
    assert res.rotation_deg == pytest.approx(expect, abs=0.3)
    assert res.peak_margin is not None and res.peak_margin >= cfg.min_peak_margin
    assert res.method_spread_deg is not None


def test_flat_texture_refused():
    pa = polar_texture("flat")
    res = estimate_polar_registration(pa, pa, None, None, PolarRegistrationConfig())
    assert not res.valid
    assert res.failure == PolarFailureKind.LOW_TEXTURE


def test_periodic_texture_refused():
    pa = polar_texture("periodic")
    res = estimate_polar_registration(pa, np.roll(pa, 180, axis=1), None, None,
                                      PolarRegistrationConfig())
    assert not res.valid
    assert res.failure != PolarFailureKind.OK


def test_ambiguous_weak_texture_refused():
    cfg = PolarRegistrationConfig()
    pa = polar_texture("weak")
    res = estimate_polar_registration(pa, np.roll(pa, 21, axis=1), None, None, cfg)
    assert not res.valid
    assert res.failure == PolarFailureKind.LOW_ANGULAR_COVERAGE
    assert "ambigu" in res.failure_reason


def test_consistency_gate_refuses_on_forced_tolerance():
    cfg = dataclasses.replace(PolarRegistrationConfig(), max_method_spread_deg=1e-9)
    pa = polar_texture("unique")
    res = estimate_polar_registration(pa, np.roll(pa, 21, axis=1), None, None, cfg)
    assert not res.valid
    assert res.failure == PolarFailureKind.NONE
    assert "disagree" in res.failure_reason


def test_invalid_geometry_correlations_refused():
    res = estimate_polar_registration(None, polar_texture("unique"), None, None,
                                      PolarRegistrationConfig())
    assert not res.valid
    empty = np.empty((0, 0), dtype=np.uint8)
    res2 = estimate_polar_registration(empty, empty, None, None,
                                       PolarRegistrationConfig())
    assert not res2.valid


def test_texture_gate_trip_via_config():
    pa = polar_texture("unique")
    cfg = dataclasses.replace(PolarRegistrationConfig(), min_texture_gradient=1.0)
    # at least verify the gate threshold is actually applied below the value
    hi = dataclasses.replace(PolarRegistrationConfig(), min_texture_gradient=10 ** 9)
    res = estimate_polar_registration(pa, pa, None, None, hi)
    assert not res.valid
    assert res.failure == PolarFailureKind.LOW_TEXTURE
    res2 = estimate_polar_registration(pa, pa, None, None, cfg)
    assert res2.valid


# ── coverage gate ──────────────────────────────────────────────────────

def test_low_angular_coverage_refused():
    pa = polar_texture("unique")
    pu = np.ones(pa.shape, dtype=bool)
    pu[:, : W // 2] = False  # only half the columns usable
    cfg = dataclasses.replace(PolarRegistrationConfig(),
                              min_angular_coverage_frac=0.6)
    res = estimate_polar_registration(pa, pa, pu, pu, cfg)
    assert not res.valid
    assert res.failure == PolarFailureKind.LOW_ANGULAR_COVERAGE


def test_unwrap_reports_mask_coverage(fs_a, no_lid, eye_src):
    gray = np.mean(eye_src, axis=2).astype(np.float32)
    usable = no_lid.build(eye_src, fs_a.roi)
    _, pu, stats = unwrap_iris(gray, fs_a.roi, 360, 64, usable)
    assert pu is not None and pu.shape == (64, 360)
    assert 0.0 < stats["angular_coverage"] <= 1.0
    assert np.any(stats["mean_column_valid"] > 0.0)


# ── end to end (masking + ROI plumbing) ────────────────────────────────

def test_end_to_end_recovers_known_rotation(eye_src, fs_a, no_lid, rot_cfg):
    _check_rotation(eye_src, fs_a, rot_cfg, no_lid, applied=3.0)


def test_end_to_end_recovers_negative_rotation(eye_src, fs_a, no_lid, rot_cfg):
    _check_rotation(eye_src, fs_a, rot_cfg, no_lid, applied=-3.0)


def test_end_to_end_zero_rotation(eye_src, fs_a, no_lid, rot_cfg):
    _check_rotation(eye_src, fs_a, rot_cfg, no_lid, applied=0.0, tol=0.6)


def _check_rotation(src, fs_a, cfg, no_lid, applied, tol=0.8):
    pe, le = iris_geometry()
    pair = make_synthetic_pair(src, PairConfig(rotation_deg=applied, center=(160, 160)))
    fs_b = detect_iris_features(pair.image_b, pe, le, config=IrisConfig()).feature_set
    res = estimate_iris_rotation(src, pair.image_b, fs_a, fs_b, cfg,
                                 masking_a=no_lid, masking_b=no_lid)
    assert res.valid, res.failure_reason
    assert res.rotation_deg == pytest.approx(applied, abs=tol)


def test_end_to_end_reflection_excluded_and_recovers(eye_src, fs_a, no_lid, rot_cfg):
    from pupil_tracking.iris import robustness as R
    from pupil_tracking.iris.paired import _post_warp_perturbation
    pe, le = iris_geometry()
    pair = make_synthetic_pair(
        eye_src, PairConfig(rotation_deg=3.0, center=(160, 160)))
    b_ref, _ = R.perturb_reflection(pair.image_b, 30, seed=5)
    fs_b = detect_iris_features(b_ref, pe, le, config=IrisConfig()).feature_set
    res = estimate_iris_rotation(eye_src, b_ref, fs_a, fs_b, rot_cfg,
                                 masking_a=no_lid, masking_b=no_lid)
    assert res.valid, res.failure_reason
    assert res.usable_fraction_b < res.usable_fraction_a  # glare excluded
    assert res.rotation_deg == pytest.approx(3.0, abs=1.5)


def test_end_to_end_low_usable_area_refused(eye_src, fs_a, no_lid, rot_cfg):
    cfg = dataclasses.replace(rot_cfg, min_usable_fraction=0.9)
    res = estimate_iris_rotation(eye_src, eye_src, fs_a, fs_a, cfg,
                                 masking_a=no_lid, masking_b=no_lid)
    assert not res.valid
    assert res.failure == PolarFailureKind.LOW_USABLE_AREA


def test_detection_disabled_by_default(eye_src, fs_a):
    res = detect_iris_rotation(eye_src, eye_src, fs_a, fs_a)
    assert isinstance(res, PolarRegistrationResult)
    assert not res.valid
    assert res.failure == PolarFailureKind.DISABLED


def test_determinism(eye_src, fs_a, no_lid, rot_cfg):
    r1 = estimate_iris_rotation(eye_src, eye_src, fs_a, fs_a, rot_cfg,
                                masking_a=no_lid, masking_b=no_lid)
    r2 = estimate_iris_rotation(eye_src, eye_src, fs_a, fs_a, rot_cfg,
                                masking_a=no_lid, masking_b=no_lid)
    assert r1.rotation_deg == r2.rotation_deg
    d1, d2 = r1.to_dict(), r2.to_dict()
    d1.pop("processing_time_ms", None)
    d2.pop("processing_time_ms", None)
    assert d1 == d2
    assert "rotation_deg" in r1.to_dict()
    assert "valid" in r1.to_dict()


def test_gradient_and_norm_channels():
    pa = polar_texture("unique")
    n = normalize_polar(pa, 15.0)
    assert n.shape == pa.shape and n.dtype == np.uint8
    g = gradient_magnitude(n)
    assert g.shape == pa.shape and g.dtype == np.uint8
    assert float(g.mean()) > 5.0  # strongly textured, above the default floor
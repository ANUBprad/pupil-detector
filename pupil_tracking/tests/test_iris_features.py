"""Tests for the Phase I iris-feature concept model.

These tests are deterministic and use synthetic fixtures. They validate the
ROI construction, masking, normalization, extraction, quality filtering and
result contracts without requiring the production ML model or clinical data
(the clinical proxy smoke test lives in ``scripts/iris_feature_smoke.py``).
"""

import numpy as np
import pytest

from pupil_tracking.iris import (
    IrisConfig,
    IrisDetectionResult,
    IrisFeatureDetector,
    IrisMasking,
    IrisNormalizer,
    IrisROIExtractor,
    detect_iris_features,
    draw_iris_overlay,
)
from pupil_tracking.iris.roi import point_in_roi_annulus, sample_annulus_mask
from pupil_tracking.iris.types import IrisFeatureType, IrisStatus
from pupil_tracking.utils.types import EllipseParams


# ── fixtures ───────────────────────────────────────────────────────────

def _make_ellipse(cx, cy, smaj, smin, angle=0.0, detected=True):
    e = EllipseParams(
        center_x=cx, center_y=cy,
        semi_major=smaj, semi_minor=smin,
        angle_deg=angle,
    )
    return e


def _synthetic_iris_image(size=320, rng_seed=0, texture=12.0, crypts=True):
    """Grayscale synthetic eye: dark pupil, brighter textured iris annulus."""
    gray = np.full((size, size), 25, np.uint8)
    center = size // 2
    cv2_circle_reuse(gray, (center, center), 130, 80)   # iris disc
    cv2_circle_reuse(gray, (center, center), 55, 10)     # pupil
    rng = np.random.default_rng(rng_seed)
    noise = rng.integers(-int(texture), int(texture), (size, size)).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if crypts:
        for a in range(0, 360, 30):
            ang = np.radians(a)
            r = 92
            x = int(center + r * np.cos(ang))
            y = int(center + r * np.sin(ang))
            cv2_circle_reuse(gray, (x, y), 4, 18)
    return gray


def cv2_circle_reuse(img, center, radius, color):
    import cv2
    cv2.circle(img, center, radius, color, -1)


def _default_geometry(size=320):
    c = size // 2
    return _make_ellipse(c, c, 55, 55), _make_ellipse(c, c, 130, 130)


# ── ROI construction ───────────────────────────────────────────────────

def test_valid_geometry_creates_valid_roi():
    pupil, limbus = _default_geometry()
    extractor = IrisROIExtractor()
    roi = extractor.build(pupil, limbus)
    assert roi.valid is True
    assert roi.reason == "ok"
    assert roi.limbus_radius_px > roi.pupil_radius_px


def test_missing_geometry_invalid_roi_no_crash():
    extractor = IrisROIExtractor()
    roi = extractor.build(None, None)
    assert roi.valid is False
    assert roi.reason  # non-empty reason


def test_missing_limbus_invalid_roi():
    pupil, _ = _default_geometry()
    extractor = IrisROIExtractor()
    assert extractor.build(pupil, None).valid is False


def test_implausible_ratio_invalid_roi():
    _, limbus = _default_geometry()
    rogue = _make_ellipse(160, 160, 125, 125)  # pupil nearly as big as limbus
    extractor = IrisROIExtractor()
    roi = extractor.build(rogue, limbus)
    assert roi.valid is False


def test_detect_with_missing_geometry_is_safe():
    img = _synthetic_iris_image()
    res = detect_iris_features(img, None, None)
    assert isinstance(res, IrisDetectionResult)
    assert res.valid is False
    assert res.status == IrisStatus.NO_ROI


def test_annulus_mask_respects_pupil_limbus():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    mask = sample_annulus_mask(img.shape, roi)
    h, w = img.shape[:2]
    # center (pupil) should not be in annulus
    assert not mask[h // 2, w // 2]
    # a ring point roughly mid-iris should be included
    c = h // 2
    r = 92
    assert mask[int(c + r * np.cos(0)), int(c + r * np.sin(0))]


# ── masking / reflection ───────────────────────────────────────────────

def test_reflection_mask_respected():
    img = _synthetic_iris_image(rng_seed=3, crypts=False)
    pupil, limbus = _default_geometry()
    detector = IrisFeatureDetector()

    # Without reflection: measure usable fraction.
    roi = detector.roi_extractor.build(pupil, limbus)
    usable_clean = detector.masking.build(img, roi)
    n_clean = int(np.count_nonzero(usable_clean))
    assert n_clean > 0

    # Add a large bright specular reflection inside the annulus.
    img2 = img.copy()
    import cv2
    c = img.shape[0] // 2
    cv2.circle(img2, (int(c + 92), c), 12, (255, 255, 255), -1)
    usable_withrefl = detector.masking.build(img2, roi)
    n_withrefl = int(np.count_nonzero(usable_withrefl))

    # A bright blob inside the annulus must reduce the usable area (the
    # reflection is removed from the mask), and never increase it.
    assert n_withrefl < n_clean
    # Some iris pixels must still be usable.
    assert n_withrefl > 0


def test_mask_stats_interface():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    detector = IrisFeatureDetector()
    roi = detector.roi_extractor.build(pupil, limbus)
    usable = detector.masking.build(img, roi)
    stats = detector.detect(img, pupil, limbus).mask_stats
    assert "usable_fraction" in stats
    assert stats["usable_fraction"] >= 0.0


# ── normalization ──────────────────────────────────────────────────────

def test_normalized_coordinates_within_bounds():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    norm = IrisNormalizer()
    a, rn = norm.to_iris_relative(float(roi.center_x + 92.0), float(roi.center_y), roi)
    assert a is not None and rn is not None
    assert 0.0 <= a < 360.0
    assert 0.0 < rn <= 1.0


def test_roundtrip_from_iris_relative():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    norm = IrisNormalizer()
    for rn in (0.0, 0.5, 1.0):
        pt = norm.from_iris_relative(35.0, rn, roi)
        assert pt is not None
        a, rn2 = norm.to_iris_relative(pt[0], pt[1], roi)
        assert rn2 is not None
        assert abs(rn2 - rn) < 0.02


# ── extraction ─────────────────────────────────────────────────────────

def test_extraction_deterministic():
    img = _synthetic_iris_image(rng_seed=5)
    pupil, limbus = _default_geometry()
    r1 = detect_iris_features(img, pupil, limbus).feature_set
    r2 = detect_iris_features(img, pupil, limbus).feature_set
    assert len(r1.features) == len(r2.features)
    for a, b in zip(r1.features, r2.features):
        assert abs(a.x - b.x) < 1e-6
        assert abs(a.y - b.y) < 1e-6
        assert a.confidence == b.confidence


def test_visibility_reflects_local_occlusion_fraction():
    # The per-feature visibility must measure the usable fraction of the
    # feature's local patch, not be a constant 1.0.
    from pupil_tracking.iris.extraction import IrisFeatureExtractor

    ex = IrisFeatureExtractor(radius_px=5)
    mask = np.ones((21, 21), dtype=bool)
    mask[0:11, :] = False  # occlude the top ~half of the image

    frac = ex._local_visibility(mask, 10.0, 10.0)
    # Feature at (10,10): patch rows 5..15 / cols 5..15 (11x11 = 121 px), of
    # which rows 5..10 (6 rows) are occluded -> usable = 121 - 66.
    assert 0.0 < frac < 1.0

    # Fully usable -> 1.0; fully occluded center would not be accepted, but a
    # fully occluded patch is reported as 1.0 for an already-usable center.
    assert ex._local_visibility(np.ones((21, 21), dtype=bool), 10.0, 10.0) == 1.0


def test_classify_uses_true_center_patch():
    # A dark pit exactly at the patch center must be classified as a crypt.
    # Previously the "center" window used a hard-coded top-left offset
    # (patch[1:4, 1:4]) that missed the true center for radius_px=5.
    from pupil_tracking.iris.extraction import IrisFeatureExtractor

    ex = IrisFeatureExtractor(radius_px=5)
    patch = np.full((11, 11), 200.0, np.float32)
    patch[5, 5] = 0.0
    assert ex._classify(patch) == IrisFeatureType.CRYPT


def test_quality_filtering_flat_iris_yields_fewer():
    # High-texture iris -> more accepted than flat (low-texture) iris.
    hi = _synthetic_iris_image(rng_seed=1, texture=18.0, crypts=True)
    lo = _synthetic_iris_image(rng_seed=1, texture=1.0, crypts=False)
    pupil, limbus = _default_geometry()
    hi_res = detect_iris_features(hi, pupil, limbus).feature_set
    lo_res = detect_iris_features(lo, pupil, limbus).feature_set
    assert hi_res.num_accepted >= lo_res.num_accepted


def test_angular_suppression_enforces_min_separator():
    img = _synthetic_iris_image(rng_seed=2)
    pupil, limbus = _default_geometry()
    fs = detect_iris_features(img, pupil, limbus).feature_set
    acc = fs.features
    if len(acc) <= 1:
        pytest.skip("not enough accepted features to verify angular separation")
    min_angular_sep_deg = IrisConfig().min_angular_sep_deg
    # Every pair of accepted features must be separated by at least the
    # configured minimum angular gap.
    for i in range(len(acc)):
        for j in range(i + 1, len(acc)):
            gap = abs(acc[i].angle_deg - acc[j].angle_deg) % 360.0
            gap = min(gap, 360.0 - gap)
            assert gap >= min_angular_sep_deg - 1e-6, (
                acc[i].angle_deg, acc[j].angle_deg, gap, min_angular_sep_deg
            )


def test_result_contract_stable():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus)
    d = res.to_dict()
    assert d["status"] == res.status.value
    assert "feature_set" in d
    feat = res.feature_set.features[0]
    fd = feat.to_dict()
    for key in ("x", "y", "angle_deg", "radial_norm", "confidence",
                "feature_type", "local_contrast", "visibility", "valid"):
        assert key in fd


def test_does_not_modify_pupil_limbus_geometry():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    p_before = (pupil.center_x, pupil.center_y, pupil.semi_major, pupil.semi_minor)
    l_before = (limbus.center_x, limbus.center_y, limbus.semi_major, limbus.semi_minor)
    detect_iris_features(img, pupil, limbus)
    p_after = (pupil.center_x, pupil.center_y, pupil.semi_major, pupil.semi_minor)
    l_after = (limbus.center_x, limbus.center_y, limbus.semi_major, limbus.semi_minor)
    assert p_before == p_after
    assert l_before == l_after


# ── visualization ──────────────────────────────────────────────────────

def test_visualization_returns_same_shape():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus)
    out = draw_iris_overlay(
        cv2_from_gray(img), res, pupil=pupil, limbus=limbus
    )
    assert out.shape == cv2_from_gray(img).shape
    assert out.dtype == np.uint8


def cv2_from_gray(gray):
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

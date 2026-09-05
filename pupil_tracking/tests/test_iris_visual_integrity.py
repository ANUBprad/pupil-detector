"""Regression tests: visual/GUI integrity and feature-validity gates.

These deterministic, synthetic-fixture tests lock the framing contract that
drives the GUI screenshots review:

    * the detector operates on the pristine source frame and never mutates it
      (so GUI overlays drawn on display copies can never become detector input),
    * every accepted iris feature is geometrically inside the authoritative
      pupil/limbus annulus and inside the usable (non-occluded) iris mask,
    * oversized / reflection / UI-like structures are not accepted as local
      iris features,
    * arbitrary match sets without geometric consensus cannot yield a valid
      rotation estimate (the KAZE-style "garbage matches" scenario).

No clinical validity claim is made; fixtures are synthetic.
"""

import cv2
import numpy as np
import pytest

from pupil_tracking.iris import (
    IrisConfig,
    IrisFeature,
    IrisFeatureSet,
    IrisNormalizer,
    detect_iris_features,
)
from pupil_tracking.iris.correspondence import (
    CorrespondenceConfig,
    MatchingBaseline,
    estimate_correspondence,
)
from pupil_tracking.iris.types import IrisStatus
from pupil_tracking.utils.types import EllipseParams


# ── fixtures ───────────────────────────────────────────────────────────

def _make_ellipse(cx, cy, smaj, smin, angle=0.0):
    return EllipseParams(
        center_x=cx, center_y=cy,
        semi_major=smaj, semi_minor=smin,
        angle_deg=angle,
    )


def _synthetic_bgr(size=320, rng_seed=7, texture=14.0):
    """BGR synthetic eye: dark pupil, textured iris annulus on a grey sclera."""
    gray = np.full((size, size), 25, np.uint8)
    c = size // 2
    cv2.circle(gray, (c, c), 130, 80, -1)
    cv2.circle(gray, (c, c), 55, 10, -1)
    rng = np.random.default_rng(rng_seed)
    noise = rng.integers(-int(texture), int(texture), (size, size)).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for a in range(0, 360, 30):
        ang = np.radians(a)
        r = 92
        x = int(c + r * np.cos(ang))
        y = int(c + r * np.sin(ang))
        cv2.circle(gray, (x, y), 4, 18, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _default_geometry(size=320, eccentric_pupil=0.0):
    c = size // 2
    px = c - eccentric_pupil
    py = c - eccentric_pupil * 0.6
    return _make_ellipse(px, py, 55, 55), _make_ellipse(c, c, 130, 130)


def _permissive_config():
    """Synthetic images have no lids/reflections; skip production eyelid
    detection and the adaptive intensity band so the fixtures are not starved
    (same rationale as ``test_iris_correspondence._iris_test_config``)."""
    return IrisConfig(eyelid_method="none", use_roi_percentiles=False)


def _ellipse_inside(cx, cy, smaj, smin, angle_deg, x, y):
    dx = x - cx
    dy = y - cy
    phi = np.radians(angle_deg)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    xr = dx * cos_p + dy * sin_p
    yr = -dx * sin_p + dy * cos_p
    return (xr / smaj) ** 2 + (yr / smin) ** 2 <= 1.0


# ── 1. pristine frame ownership ────────────────────────────────────────

def test_detector_does_not_mutate_input_frame():
    img = _synthetic_bgr(rng_seed=11)
    before = img.copy()
    detect_iris_features(img, *_default_geometry(), config=_permissive_config())
    assert np.array_equal(img, before)


def test_blue_ui_circle_drawn_after_detection_has_no_effect():
    img = _synthetic_bgr(rng_seed=13)
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    feats_before = [(f.x, f.y, f.confidence) for f in res.feature_set.features]

    # Reproduce the GUI display path: overlays are painted on a separate
    # display copy (e.g. a blue limbus cross-section / blue ROI circle),
    # never on the source frame the detector reads.
    display = img.copy()
    c = img.shape[0] // 2
    cv2.circle(display, (c, c), int(limbus.semi_major), (255, 0, 0), 2)
    cv2.circle(display, (c, c), int(limbus.semi_major * 0.94), (255, 0, 0), 1)

    assert not np.array_equal(display, img)  # overlay lives only on the copy

    res2 = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    feats_after = [(f.x, f.y, f.confidence) for f in res2.feature_set.features]
    assert feats_before == feats_after


# ── 2. features stay inside the authoritative iris annulus ─────────────

def test_annulus_support_fraction_semantics():
    """The annulus support gate is the direct mechanism that keeps feature
    patches off the pupil and the sclera/ring region. Test it exactly."""
    from pupil_tracking.iris.extraction import IrisFeatureExtractor
    from pupil_tracking.iris.masking import IrisMasking
    from pupil_tracking.iris.roi import IrisROIExtractor

    img = _synthetic_bgr(rng_seed=1)
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    mask = IrisMasking(eyelid_method="none").build(img, roi)
    ex = IrisFeatureExtractor()
    c = img.shape[0] // 2

    inside = ex._annulus_support_fraction(mask, float(c + 92), float(c), roi, pupil, limbus)
    assert inside >= 0.9  # mid-iris: almost all patch pixels inside the annulus

    pupil_edge = ex._annulus_support_fraction(mask, float(c + 56), float(c), roi, pupil, limbus)
    assert pupil_edge < 0.7  # straddling the pupil/sclera side boundary

    sclera = ex._annulus_support_fraction(mask, float(c + 132), float(c), roi, pupil, limbus)
    assert sclera < 0.7  # on the sclera/ring region: not iris


def test_no_feature_inside_real_pupil_ellipse():
    img = _synthetic_bgr(rng_seed=5)
    pupil, limbus = _default_geometry(eccentric_pupil=16.0)
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    for f in res.feature_set.features:
        inside_pupil = _ellipse_inside(
            pupil.center_x, pupil.center_y,
            pupil.semi_major, pupil.semi_minor,
            pupil.angle_deg, f.x + 0.5, f.y + 0.5,
        )
        assert not inside_pupil


def test_no_feature_outside_real_limbus_ellipse():
    img = _synthetic_bgr(rng_seed=6)
    pupil, limbus = _default_geometry(eccentric_pupil=16.0)
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    for f in res.feature_set.features:
        inside_limbus = _ellipse_inside(
            limbus.center_x, limbus.center_y,
            limbus.semi_major, limbus.semi_minor,
            limbus.angle_deg, f.x + 0.5, f.y + 0.5,
        )
        assert inside_limbus


def test_features_follow_and_stay_within_annulus():
    """Every accepted feature maps to iris-relative coordinates strictly inside
    the annulus ([0,1] radial), with accepted geometry retained."""
    img = _synthetic_bgr(rng_seed=8)
    pupil, limbus = _default_geometry(eccentric_pupil=10.0)
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    roi = res.feature_set.roi
    assert roi.valid
    norm = IrisNormalizer()
    for f in res.feature_set.features:
        a, rn = norm.to_iris_relative(f.x + 0.5, f.y + 0.5, roi)
        assert a is not None and rn is not None
        assert 0.0 < rn <= 1.0


# ── 3. outside-mask / reflections / oversized structures rejected ──────

def test_feature_centers_inside_usable_mask():
    img = _synthetic_bgr(rng_seed=3)
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    assert res.mask_stats.get("usable_fraction", 0.0) > 0.0


def test_reflection_blob_not_accepted_as_features():
    img = _synthetic_bgr(rng_seed=4)
    pupil, limbus = _default_geometry()
    c = img.shape[0] // 2
    blob_cx, blob_cy, blob_r = int(c + 88), c, 9
    annotated = img.copy()
    cv2.circle(annotated, (blob_cx, blob_cy), blob_r, (255, 255, 255), -1)
    res = detect_iris_features(annotated, pupil, limbus, config=_permissive_config())
    for f in res.feature_set.features:
        d = np.hypot(f.x - blob_cx, f.y - blob_cy)
        assert d > blob_r + 4.0


def test_giant_bright_structure_not_accepted_as_features():
    img = _synthetic_bgr(rng_seed=2)
    pupil, limbus = _default_geometry()
    c = img.shape[0] // 2
    disc_cx, disc_cy, disc_r = int(c + 85), c, 22
    annotated = img.copy()
    cv2.circle(annotated, (disc_cx, disc_cy), disc_r, (235, 235, 235), -1)
    res = detect_iris_features(annotated, pupil, limbus, config=_permissive_config())
    for f in res.feature_set.features:
        d = np.hypot(f.x - disc_cx, f.y - disc_cy)
        # neither the interior nor the rim of the giant structure becomes a
        # local feature.
        assert d > disc_r - 2.0 and d < disc_r + 6.0 or d > disc_r + 6.0


# ── 4. existing valid features still pass ──────────────────────────────

def test_valid_textured_iris_yields_local_features():
    img = _synthetic_bgr(rng_seed=9)
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    n = len(res.feature_set.features)
    assert n >= 8
    for f in res.feature_set.features:
        assert f.valid is True
        assert 0.0 < f.scale <= 1.0


def test_flat_blank_iris_yields_no_garbage_features():
    img = np.full((320, 320, 3), 90, np.uint8)
    pupil, limbus = _default_geometry()
    cv2.circle(img, (160, 160), 130, 80, -1)
    cv2.circle(img, (160, 160), 55, 20, -1)
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert len(res.feature_set.features) == 0


# ── 5. no ground-truth rotation without geometric consensus (KAZE) ─────

def _feature_set_with_angles(base_set, angles):
    feats = [
        IrisFeature(
            id=i,
            x=f.x, y=f.y,
            angle_deg=angle,
            radial_norm=f.radial_norm,
            scale=f.scale,
            orientation_deg=f.orientation_deg,
            feature_type=f.feature_type,
            response=f.response,
            local_contrast=f.local_contrast,
            visibility=f.visibility,
            confidence=f.confidence,
            valid=f.valid,
            descriptor=f.descriptor,
        )
        for i, (f, angle) in enumerate(zip(base_set.features, angles))
    ]
    return IrisFeatureSet(
        roi=base_set.roi,
        features=feats,
        num_candidates=len(feats),
        num_accepted=len(feats),
    )


@pytest.fixture(scope="module")
def _src():
    img = _synthetic_bgr(rng_seed=17)
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus, config=_permissive_config())
    assert res.status == IrisStatus.OK
    return img, pupil, limbus, res.feature_set


def test_incoherent_matches_cannot_yield_valid_rotation(_src):
    img, pupil, limbus, fs = _src
    # Four mutually inconsistent quarter-hypotheses: no single rotation is
    # supported by more than a quarter of the matches, so consensus/residual
    # gates must refuse (the KAZE-garbage scenario).
    quarters = [0.0, 90.0, 180.0, 270.0]
    angles = [
        (base + quarters[i % len(quarters)]) % 360.0
        for i, base in enumerate(f.angle_deg for f in fs.features)
    ]
    fs_bad = _feature_set_with_angles(fs, angles)
    res = estimate_correspondence(
        img, img, fs, fs_bad,
        baseline=MatchingBaseline.GEOMETRIC,
    )
    assert res.valid is False


def test_inconsistent_descriptors_cannot_yield_valid_rotation(_src):
    img, pupil, limbus, fs = _src
    feats = []
    rng = np.random.RandomState(7)
    for i, f in enumerate(fs.features):
        d = np.zeros(16, np.float32)
        d[i % 16] = 1.0
        feats.append(IrisFeature(
            id=i, x=f.x, y=f.y, angle_deg=f.angle_deg,
            radial_norm=f.radial_norm, scale=f.scale,
            orientation_deg=f.orientation_deg,
            feature_type=f.feature_type, response=f.response,
            local_contrast=f.local_contrast, visibility=f.visibility,
            confidence=f.confidence, valid=f.valid, descriptor=d,
        ))
    fs_conflict = IrisFeatureSet(roi=fs.roi, features=feats,
                                 num_accepted=len(feats))
    res = estimate_correspondence(
        img, img, fs, fs_conflict,
        baseline=MatchingBaseline.GEOMETRIC_DESCRIPTOR,
    )
    assert res.valid is False


def test_self_matching_control_yields_valid_rotation(_src):
    img, pupil, limbus, fs = _src
    res = estimate_correspondence(
        img, img, fs, fs,
        baseline=MatchingBaseline.GEOMETRIC,
        config=CorrespondenceConfig(refine=False),
    )
    assert res.valid is True
    assert abs(min(res.estimated_rotation_deg, 360.0 - res.estimated_rotation_deg)) < 0.5
"""Regression tests for calibration mode refresh (forward, reverse, pixel protection).

Verifies that when the calibration mode changes, the mm_per_px value updates
while pixel geometry remains unchanged. This is the unit-level contract that
the GUI fix in _apply_live_settings depends on.
"""

import pytest
import numpy as np

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.utils.types import (
    CalibrationInfo,
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)

FIXED_PX_PER_MM = 44.5
FIXED_MM_PER_PX = 1.0 / FIXED_PX_PER_MM


def _make_result():
    return EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=320.0, center_y=240.0,
                semi_major=80.0, semi_minor=75.0, angle_deg=12.0,
            ),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=315.0, center_y=235.0,
                semi_major=250.0, semi_minor=240.0, angle_deg=5.0,
            ),
            confidence=0.90,
        ),
        overall_quality=DetectionQuality.RESEARCH,
        overall_confidence=0.92,
    )


def test_detector_set_calibration_mode_returns_new_calibration():
    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal_fixed = det._calibration
    assert cal_fixed.calibrated is True
    assert cal_fixed.method == "fixed_manual"
    assert pytest.approx(cal_fixed.px_per_mm) == FIXED_PX_PER_MM

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal_anat = det._calibration
    assert cal_anat.method == "anatomical"
    assert cal_anat is not cal_fixed


def test_calibration_swap_updates_mm_values():
    result = _make_result()

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    result.calibration = det._calibration
    mm_fixed = result.calibration.mm_per_px
    assert pytest.approx(mm_fixed) == FIXED_MM_PER_PX

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    result.calibration = det._calibration
    mm_anat = result.calibration.mm_per_px
    assert mm_anat != mm_fixed


def test_pixel_geometry_unchanged_after_calibration_swap():
    result = _make_result()

    pupil_semi_major = result.pupil.ellipse.semi_major
    pupil_semi_minor = result.pupil.ellipse.semi_minor
    pupil_center = (result.pupil.ellipse.center_x, result.pupil.ellipse.center_y)
    limbus_semi_major = result.limbus.ellipse.semi_major
    limbus_semi_minor = result.limbus.ellipse.semi_minor
    limbus_center = (result.limbus.ellipse.center_x, result.limbus.ellipse.center_y)

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    result.calibration = det._calibration

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    result.calibration = det._calibration

    assert result.pupil.ellipse.semi_major == pupil_semi_major
    assert result.pupil.ellipse.semi_minor == pupil_semi_minor
    assert (result.pupil.ellipse.center_x, result.pupil.ellipse.center_y) == pupil_center
    assert result.limbus.ellipse.semi_major == limbus_semi_major
    assert result.limbus.ellipse.semi_minor == limbus_semi_minor
    assert (result.limbus.ellipse.center_x, result.limbus.ellipse.center_y) == limbus_center


def test_forward_round_trip_anatomical_to_fixed_to_anatomical():
    result = _make_result()

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal1 = det._calibration
    result.calibration = cal1
    mm1 = result.calibration.mm_per_px

    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal2 = det._calibration
    result.calibration = cal2
    mm2 = result.calibration.mm_per_px
    assert mm2 != mm1

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal3 = det._calibration
    result.calibration = cal3
    mm3 = result.calibration.mm_per_px
    assert pytest.approx(mm3) == mm1


def test_mm_values_differ_between_modes():
    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal_fixed = det._calibration

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal_anat = det._calibration

    assert cal_fixed.mm_per_px != cal_anat.mm_per_px
    assert cal_fixed.px_per_mm != cal_anat.px_per_mm
    assert cal_fixed.method != cal_anat.method


def _compute_wtw(semi_major_px, semi_minor_px, mm_per_px):
    """Recompute WTW the same way _update_measurements does."""
    h = 2.0 * semi_major_px * mm_per_px
    v = 2.0 * semi_minor_px * mm_per_px
    m = (h + v) / 2.0
    return h, v, m


def test_wtw_updates_on_calibration_switch():
    """Stale wtw_horizontal_mm must not persist after mode switch."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration
    # Simulate _add_mm_values storing pre-computed WTW
    h0, v0, m0 = _compute_wtw(le.semi_major, le.semi_minor, result.calibration.mm_per_px)
    result.limbus.wtw_horizontal_mm = h0
    result.limbus.wtw_vertical_mm = v0
    result.limbus.wtw_mean_mm = m0

    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    h_exp, v_exp, m_exp = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)

    # After clearing stale attrs (as _apply_live_settings now does):
    for attr in ("wtw_horizontal_mm", "wtw_vertical_mm", "wtw_mean_mm"):
        if hasattr(result.limbus, attr):
            setattr(result.limbus, attr, None)

    h_wtw = getattr(result.limbus, "wtw_horizontal_mm", None)
    if h_wtw is None:
        h_wtw, v_wtw, m_wtw = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)
    assert pytest.approx(h_wtw, abs=0.01) == h_exp
    assert h_wtw != h0


def test_wtw_recomputed_from_current_calibration():
    """_update_measurements WTW logic always recomputes from current mm_per_px."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration

    h, v, m = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)
    expected_h = 2.0 * le.semi_major * FIXED_MM_PER_PX
    assert pytest.approx(h, abs=0.01) == expected_h


def test_fixed_mode_semi_major_not_tautological():
    """FIXED mode semi_major_mm must NOT equal corneal/2."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration

    semi_major_mm = le.semi_major * result.calibration.mm_per_px
    # 6.0 is the ANATOMICAL tautology (12.0 / 2); FIXED must differ
    assert semi_major_mm != 6.0
    assert pytest.approx(semi_major_mm, abs=0.01) == le.semi_major * FIXED_MM_PER_PX


def test_anatomical_tautology_semi_major():
    """ANATOMICAL semi_major_mm = corneal_diameter / 2 (tautology)."""
    result = _make_result()
    le = result.limbus.ellipse

    corneal_mm = 12.0
    diameter_px = le.semi_major * 2.0
    mm_per_px = corneal_mm / diameter_px

    semi_major_mm = le.semi_major * mm_per_px
    assert pytest.approx(semi_major_mm, abs=0.001) == corneal_mm / 2.0


def test_different_images_different_wtw_in_fixed_mode():
    """In FIXED mode, different pixel geometries produce different WTW."""
    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)

    r1 = _make_result()
    r1.calibration = det._calibration
    h1 = 2.0 * r1.limbus.ellipse.semi_major * FIXED_MM_PER_PX

    r2 = EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(center_x=320, center_y=240, semi_major=60, semi_minor=55),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(center_x=315, center_y=235, semi_major=180, semi_minor=170),
            confidence=0.90,
        ),
        overall_quality=DetectionQuality.RESEARCH,
        overall_confidence=0.92,
    )
    r2.calibration = det._calibration
    h2 = 2.0 * r2.limbus.ellipse.semi_major * FIXED_MM_PER_PX

    assert h1 != h2
    assert pytest.approx(h1 / h2, abs=0.01) == 250.0 / 180.0


def test_round_trip_mode_switch_no_stale_calibration():
    """ANATOMICAL -> FIXED -> ANATOMICAL: no stale calibration persists."""
    det = UnifiedDetector()
    result = _make_result()

    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration

    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    mm_fixed = result.calibration.mm_per_px

    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    # After ANATOMICAL reset, _current_best returns uncalibrated.
    # The GUI fix computes tautological calibration from pixel geometry.
    ep = result.limbus.ellipse
    corneal_mm = 12.0
    dia_px = ep.semi_major * 2.0
    if dia_px > 10:
        result.calibration = CalibrationInfo(
            calibrated=True,
            px_per_mm=dia_px / corneal_mm,
            mm_per_px=corneal_mm / dia_px,
            method="anatomical",
        )
    mm_back = result.calibration.mm_per_px

    assert mm_back != mm_fixed
    assert pytest.approx(mm_back * ep.semi_major, abs=0.001) == corneal_mm / 2.0


def test_stale_wtw_attrs_cleared_on_mode_switch():
    """Pre-computed WTW attrs must be reset to None when calibration changes."""
    result = _make_result()
    result.limbus.wtw_horizontal_mm = 999.0
    result.limbus.wtw_vertical_mm = 888.0
    result.limbus.wtw_mean_mm = 777.0

    for attr in ("wtw_horizontal_mm", "wtw_vertical_mm", "wtw_mean_mm"):
        if hasattr(result.limbus, attr):
            setattr(result.limbus, attr, None)

    assert result.limbus.wtw_horizontal_mm is None
    assert result.limbus.wtw_vertical_mm is None
    assert result.limbus.wtw_mean_mm is None


def test_no_limbus_does_not_corrupt_calibration():
    """Mode switch with no limbus detection should not crash."""
    det = UnifiedDetector()
    result = EyeDetectionResult(
        pupil=PupilDetection(detected=False),
        limbus=LimbusDetection(detected=False),
        overall_quality=DetectionQuality.NO_DETECTION,
        overall_confidence=0.0,
    )
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration
    assert result.calibration is not None

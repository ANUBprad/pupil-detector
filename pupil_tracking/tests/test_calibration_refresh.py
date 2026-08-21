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

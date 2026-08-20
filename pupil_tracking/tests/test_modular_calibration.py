"""Unit tests for modular calibration, physical unit scaling, and ROI lock-on."""

import pytest
import numpy as np
from types import SimpleNamespace

from pupil_tracking.calibration.spatial_calibration import (
    SpatialCalibrator,
    StabilizedCalibrator,
)
from pupil_tracking.utils.config import CalibrationConfig, MeasurementStabilizationConfig
from pupil_tracking.utils.types import (
    CalibrationInfo,
    EllipseParams,
    LimbusDetection,
    PupilDetection,
)
from pupil_tracking.core.eye_roi_detector import EyeROIDetector


def test_anatomical_anchor_calibration():
    calibrator = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=11.5)
    limbus = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(
            center_x=320.0,
            center_y=240.0,
            semi_major=250.0,
            semi_minor=240.0,
            angle_deg=0.0,
        ),
        confidence=0.9,
    )
    cal = calibrator.calibrate_from_limbus(limbus)
    assert cal.calibrated is True
    assert cal.method == "anatomical"
    assert cal.corneal_diameter_assumed_mm == 11.5
    # Mean diameter = 250 + 240 = 490 px -> 490 / 11.5 = 42.609 px/mm
    assert pytest.approx(cal.px_per_mm, rel=1e-3) == 490.0 / 11.5
    assert pytest.approx(cal.mm_per_px, rel=1e-3) == 11.5 / 490.0


def test_fixed_pixel_scale_dynamic_limbus_mm():
    fixed_scale = 44.5  # px/mm
    calibrator = SpatialCalibrator(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=fixed_scale,
    )
    
    # Frame 1: limbus semi_major = 250 px
    limbus1 = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(
            center_x=320.0,
            center_y=240.0,
            semi_major=250.0,
            semi_minor=245.0,
        ),
        confidence=0.9,
    )
    cal1 = calibrator.calibrate_from_limbus(limbus1)
    assert cal1.calibrated is True
    assert cal1.method == "fixed_manual"
    assert cal1.corneal_diameter_assumed_mm is None
    assert pytest.approx(cal1.px_per_mm, rel=1e-4) == 44.5
    
    limbus1_semi_major_mm = limbus1.ellipse.semi_major * cal1.mm_per_px
    assert pytest.approx(limbus1_semi_major_mm, rel=1e-3) == 250.0 / 44.5  # ~5.618 mm

    # Frame 2: limbus semi_major expands to 270 px
    limbus2 = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(
            center_x=320.0,
            center_y=240.0,
            semi_major=270.0,
            semi_minor=265.0,
        ),
        confidence=0.9,
    )
    cal2 = calibrator.calibrate_from_limbus(limbus2)
    limbus2_semi_major_mm = limbus2.ellipse.semi_major * cal2.mm_per_px
    assert pytest.approx(limbus2_semi_major_mm, rel=1e-3) == 270.0 / 44.5  # ~6.067 mm

    # Crucial assertion: semi_major_mm is NOT locked to constant 5.75 mm!
    assert abs(limbus1_semi_major_mm - limbus2_semi_major_mm) > 0.4
    assert limbus1_semi_major_mm != 5.75
    assert limbus2_semi_major_mm != 5.75


def test_stabilized_calibrator_modes():
    stab_cfg = MeasurementStabilizationConfig()
    sc = StabilizedCalibrator(config=stab_cfg, mode="FIXED_PIXEL_SCALE", manual_px_per_mm=50.0)
    limbus = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=200, center_y=200, semi_major=250, semi_minor=250),
        confidence=0.95,
    )
    cal = sc.update_from_limbus(limbus)
    assert cal.calibrated is True
    assert cal.method == "fixed_manual"
    assert pytest.approx(cal.px_per_mm) == 50.0

    # Switch to RING_REFLECTION
    sc.set_mode(mode="RING_REFLECTION", ring_diameter_mm=9.4)
    cal_ring = sc.update_from_ring(ring_radius_px=235.0)
    assert cal_ring.calibrated is True
    assert cal_ring.method == "ring_reflection"
    # Dia = 470 px / 9.4 mm = 50.0 px/mm
    assert pytest.approx(cal_ring.px_per_mm) == 50.0


def test_calibration_info_serialization():
    cal = CalibrationInfo(
        calibrated=True,
        px_per_mm=44.5,
        mm_per_px=1.0 / 44.5,
        source="fixed_manual",
        method="fixed_manual",
        reference_diameter_mm=0.0,
        reference_diameter_px=0.0,
        confidence=1.0,
        corneal_diameter_assumed_mm=None,
    )
    d = cal.to_dict()
    assert d["calibrated"] is True
    assert d["method"] == "fixed_manual"
    assert d["corneal_diameter_assumed_mm"] is None
    assert pytest.approx(d["px_per_mm"]) == 44.5


def test_eye_roi_detector_closeup_lockon():
    detector = EyeROIDetector()
    # Create a synthetic closeup image (dark pupil/iris with no full face)
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    import cv2
    cv2.circle(img, (320, 240), 90, (30, 30, 30), -1)  # dark circle
    
    # Within <= 3 frames, it should recognize eye closeup or find ROI
    roi = None
    for _ in range(3):
        roi = detector.detect(img)
    assert roi is not None
    assert roi.valid is True


def test_calculate_ruler_scale():
    from pupil_tracking.calibration.spatial_calibration import calculate_ruler_scale
    p1 = (100.0, 100.0)
    p2 = (300.0, 100.0)  # 200 px distance
    known_dist_mm = 10.0
    px_per_mm, mm_per_px = calculate_ruler_scale(p1, p2, known_dist_mm)
    assert pytest.approx(px_per_mm) == 20.0
    assert pytest.approx(mm_per_px) == 0.05


def test_evaluate_clinical_wtw_fixed_scale():
    from pupil_tracking.calibration.spatial_calibration import evaluate_clinical_wtw
    
    cal = CalibrationInfo(
        calibrated=True,
        px_per_mm=44.5,
        mm_per_px=1.0 / 44.5,
        source="fixed_manual",
        method="fixed_manual",
    )
    
    # Normal Cornea: semi_major = 260px, semi_minor = 250px
    # H = 2 * 260 / 44.5 = 11.685 mm, V = 2 * 250 / 44.5 = 11.236 mm, Mean = 11.461 mm
    limbus_normal = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=260.0, semi_minor=250.0),
        confidence=0.9,
    )
    h, v, m, astig, is_m, status = evaluate_clinical_wtw(limbus_normal, cal)
    assert is_m is True
    assert status == "VALID_CLINICAL_RANGE"
    assert pytest.approx(h, rel=1e-3) == 11.685
    assert pytest.approx(v, rel=1e-3) == 11.236
    assert pytest.approx(m, rel=1e-3) == 11.461
    assert pytest.approx(astig, rel=1e-3) == 0.449

    # Microcornea: semi_major = 180px, semi_minor = 180px -> Mean = 8.09 mm (< 9.5 mm)
    limbus_micro = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=180.0, semi_minor=180.0),
        confidence=0.9,
    )
    h, v, m, astig, is_m, status = evaluate_clinical_wtw(limbus_micro, cal)
    assert is_m is True
    assert status == "OUT_OF_BOUNDS_WARNING"
    assert m < 9.5

    # Megalocornea: semi_major = 350px, semi_minor = 350px -> Mean = 15.73 mm (> 13.5 mm)
    limbus_megalo = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=350.0, semi_minor=350.0),
        confidence=0.9,
    )
    h, v, m, astig, is_m, status = evaluate_clinical_wtw(limbus_megalo, cal)
    assert is_m is True
    assert status == "OUT_OF_BOUNDS_WARNING"
    assert m > 13.5


def test_evaluate_clinical_wtw_anatomical_anchor():
    from pupil_tracking.calibration.spatial_calibration import evaluate_clinical_wtw
    
    cal_anat = CalibrationInfo(
        calibrated=True,
        px_per_mm=500.0 / 11.5,
        mm_per_px=11.5 / 500.0,
        source="limbus",
        method="anatomical",
        corneal_diameter_assumed_mm=11.5,
    )
    limbus = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=250.0, semi_minor=240.0),
        confidence=0.9,
    )
    h, v, m, astig, is_m, status = evaluate_clinical_wtw(limbus, cal_anat)
    assert is_m is False
    assert status == "ANCHORED_BASELINE"
    assert pytest.approx(h, rel=1e-3) == 11.5
    d = limbus.to_dict()
    assert "wtw_horizontal_mm" in d
    assert "wtw_validity_status" in d


# ================================================================
# Regression tests — semi-major and WTW must be dynamic
# ================================================================

def test_semi_major_mm_dynamic_anatomical_anchor():
    """Different elliptical geometries must yield different semi-major mm values.

    Before the fix, semi_major_mm was always corneal/2 due to
    calibrating from semi_major and then converting semi_major with
    the same scale.
    """
    corneal = 12.0
    calibrator = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal)

    limbus_a = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=250.0, semi_minor=240.0),
        confidence=0.9,
    )
    limbus_b = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=300.0, semi_minor=240.0),
        confidence=0.9,
    )

    cal_a = calibrator.calibrate_from_limbus(limbus_a)
    cal_b = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal).calibrate_from_limbus(limbus_b)

    smm_a = limbus_a.ellipse.semi_major * cal_a.mm_per_px
    smm_b = limbus_b.ellipse.semi_major * cal_b.mm_per_px

    assert smm_a != corneal / 2.0, "semi_major_mm must not equal corneal/2 (tautology)"
    assert smm_b != corneal / 2.0
    assert pytest.approx(smm_a, rel=1e-6) == 250.0 * 12.0 / (250.0 + 240.0)
    assert pytest.approx(smm_b, rel=1e-6) == 300.0 * 12.0 / (300.0 + 240.0)
    assert abs(smm_a - smm_b) > 0.1


def test_horizontal_wtw_dynamic_anatomical_anchor():
    """Different elliptical geometries must yield different horizontal WTW values."""
    from pupil_tracking.calibration.spatial_calibration import evaluate_clinical_wtw

    corneal = 12.0
    calibrator = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal)

    limbus_a = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=250.0, semi_minor=240.0),
        confidence=0.9,
    )
    limbus_b = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=300.0, semi_minor=240.0),
        confidence=0.9,
    )

    cal_a = calibrator.calibrate_from_limbus(limbus_a)
    cal_b = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal).calibrate_from_limbus(limbus_b)

    h_a, _, _, _, _, _ = evaluate_clinical_wtw(limbus_a, cal_a)
    h_b, _, _, _, _, _ = evaluate_clinical_wtw(limbus_b, cal_b)

    assert h_a != corneal, "h_wtw must not equal corneal (tautology)"
    assert h_b != corneal
    assert abs(h_a - h_b) > 0.1


def test_semi_minor_mm_dynamic_anatomical_anchor():
    """Different semi-minor values must yield different semi-minor mm values."""
    corneal = 12.0

    limbus_a = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=260.0, semi_minor=240.0),
        confidence=0.9,
    )
    limbus_b = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=260.0, semi_minor=280.0),
        confidence=0.9,
    )

    cal_a = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal).calibrate_from_limbus(limbus_a)
    cal_b = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=corneal).calibrate_from_limbus(limbus_b)

    smm_a = limbus_a.ellipse.semi_minor * cal_a.mm_per_px
    smm_b = limbus_b.ellipse.semi_minor * cal_b.mm_per_px

    assert smm_a != smm_b
    assert pytest.approx(smm_a, rel=1e-6) == 240.0 * 12.0 / (260.0 + 240.0)
    assert pytest.approx(smm_b, rel=1e-6) == 280.0 * 12.0 / (260.0 + 280.0)


def test_assumed_cornea_does_not_overwrite_independent():
    """With FIXED_PIXEL_SCALE, changing assumed corneal diameter must NOT change mm values."""
    fixed_scale = 44.5
    cornea_a = 11.5
    cornea_b = 12.0

    limbus = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=260.0, semi_minor=250.0),
        confidence=0.9,
    )

    cal_a = SpatialCalibrator(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=fixed_scale).calibrate_from_limbus(
        limbus, corneal_diameter_mm=cornea_a
    )
    cal_b = SpatialCalibrator(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=fixed_scale).calibrate_from_limbus(
        limbus, corneal_diameter_mm=cornea_b
    )

    smm_a = limbus.ellipse.semi_major * cal_a.mm_per_px
    smm_b = limbus.ellipse.semi_major * cal_b.mm_per_px
    assert pytest.approx(smm_a) == pytest.approx(smm_b), \
        "Independent calibration must ignore assumed corneal diameter"
    assert cal_a.corneal_diameter_assumed_mm is None
    assert cal_b.corneal_diameter_assumed_mm is None


def test_pixel_geometry_unchanged_by_calibration():
    """Changing calibration reference must never modify pixel measurements."""
    limbus = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=320, center_y=240, semi_major=260.0, semi_minor=250.0),
        confidence=0.9,
    )
    pre_major = limbus.ellipse.semi_major
    pre_minor = limbus.ellipse.semi_minor

    SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=11.5).calibrate_from_limbus(limbus)
    assert limbus.ellipse.semi_major == pre_major
    assert limbus.ellipse.semi_minor == pre_minor

    SpatialCalibrator(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=44.5).calibrate_from_limbus(limbus)
    assert limbus.ellipse.semi_major == pre_major
    assert limbus.ellipse.semi_minor == pre_minor


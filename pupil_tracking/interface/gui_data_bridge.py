# pupil_tracking/interface/gui_data_bridge.py
"""Dict-to-namespace bridge and frame result adaptation functions
extracted from :class:`PupilTrackingGUI` during the Phase-4
refactoring.

Converts flat detection dicts from OptimizedVideoProcessor /
FastInference into the SimpleNamespace objects that the GUI
measurement panels and export functions expect.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from pupil_tracking.utils.types import DetectionQuality

# ── Constants ──────────────────────────────────────────────────────

from pupil_tracking.utils.config import CORNEAL_DIAMETER_MM as _CORNEAL_DIAMETER_MM


# ── Dict → SimpleNamespace conversion ──────────────────────────────


def dict_to_frame_ns(d: dict) -> SimpleNamespace:
    """Convert a flat detection dict (from OptimizedVideoProcessor /
    FastInference / result_to_dict) into the SimpleNamespace that
    ``adapt_frame_result`` expects.
    """
    pupil_det = d.get("pupil_detected", False)
    limbus_det = d.get("limbus_detected", False)

    if pupil_det:
        px, py = d.get("pupil_x", 0.0), d.get("pupil_y", 0.0)
        pr = d.get("pupil_radius", 0.0)
        p_major = d.get("pupil_major", pr * 2)
        p_minor = d.get("pupil_minor", pr * 2)
        p_angle = d.get("pupil_angle", 0.0)
        pupil_center = (px, py)
        pupil_axes = (p_major, p_minor)
        pupil_angle = p_angle
    else:
        pupil_center = None
        pupil_axes = None
        pupil_angle = 0.0

    if limbus_det:
        lx, ly = d.get("limbus_x", 0.0), d.get("limbus_y", 0.0)
        lr = d.get("limbus_radius", 0.0)
        l_major = d.get("limbus_major", lr * 2)
        l_minor = d.get("limbus_minor", lr * 2)
        l_angle = d.get("limbus_angle", 0.0)
        limbus_center = (lx, ly)
        limbus_axes = (l_major, l_minor)
        limbus_angle = l_angle
    else:
        limbus_center = None
        limbus_axes = None
        limbus_angle = 0.0

    conf = d.get(
        "overall_confidence",
        d.get("pupil_confidence", d.get("confidence", 0.0)),
    )

    q_str = d.get("overall_quality", "")
    quality = SimpleNamespace(value=q_str) if q_str else None

    return SimpleNamespace(
        pupil_center=pupil_center,
        pupil_axes=pupil_axes,
        pupil_angle=pupil_angle,
        limbus_center=limbus_center,
        limbus_axes=limbus_axes,
        limbus_angle=limbus_angle,
        confidence=conf,
        quality=quality,
        pupil_fit_type=d.get("pupil_fit_type"),
        limbus_fit_type=d.get("limbus_fit_type"),
        processing_ms=d.get("processing_time_ms", d.get("latency_ms", 0.0)),
        latency_ms=d.get("latency_ms", d.get("processing_time_ms", 0.0)),
        frame_number=d.get("frame_idx", 0),
        is_interpolated=not d.get("pupil_detected", False),
        ring_status=d.get("ring_status", "unknown"),
        ring_center=(
            (d.get("ring_center_x"), d.get("ring_center_y"))
            if d.get("ring_center_x") is not None and d.get("ring_center_y") is not None
            else None
        ),
        ring_radius=d.get("ring_radius"),
        ring_dot_count=d.get("ring_dot_count", 0),
        corneal_reference_source=d.get("corneal_reference_source", "limbus"),
        reuse_cached_result=bool(d.get("reuse_cached_result", False)),
        reuse_reason=d.get("reuse_reason"),
        _eye_result=d.get("_eye_result"),
    )


# ── Frame result → adapted namespace ───────────────────────────────


def adapt_frame_result(
    fr: Any,
    frame_shape: Tuple[int, ...],
    frame_result_to_export_dict_fn: Any = None,
) -> SimpleNamespace:
    """Adapt a SimpleNamespace frame result into the full
    EyeDetectionResult-compatible namespace with calibration,
    corneal center, and quality.

    Parameters
    ----------
    fr : SimpleNamespace
        Raw frame result from OptimizedVideoProcessor.
    frame_shape : tuple
        Shape of the frame (H, W, ...).
    frame_result_to_export_dict_fn : callable, optional
        Reference to the export dict function (set as bound method).
    """
    H, W = frame_shape[:2]
    eye_result = getattr(fr, "_eye_result", None)
    if eye_result is not None:
        eye_result.metadata.image_width = W
        eye_result.metadata.image_height = H
        eye_result.metadata.frame_number = getattr(fr, "frame_number", 0)
        eye_result.metadata.latency_ms = getattr(fr, "latency_ms", fr.processing_ms)
        return eye_result

    if fr.limbus_axes is not None:
        limbus_semi_major_dia = float(max(fr.limbus_axes))
        px_per_mm = limbus_semi_major_dia / _CORNEAL_DIAMETER_MM
        mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 0.0
        cal = SimpleNamespace(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=mm_per_px,
            source="limbus_semi_major (optimised)",
            reference_diameter_mm=_CORNEAL_DIAMETER_MM,
            reference_diameter_px=limbus_semi_major_dia,
            confidence=min(0.95, fr.confidence + 0.05),
        )
    else:
        cal = SimpleNamespace(
            calibrated=False,
            px_per_mm=0.0,
            mm_per_px=0.0,
            source="none",
            reference_diameter_mm=0.0,
            reference_diameter_px=0.0,
            confidence=0.0,
        )

    _MAP = {
        "SURGICAL": "SURGICAL",
        "CLINICAL": "CLINICAL",
        "INTERPOLATED": "RESEARCH",
        "PREDICTED": "RESEARCH",
        "FAILED": "NO_DETECTION",
    }
    if fr.quality:
        raw_quality = fr.quality.value
        if raw_quality in _MAP:
            q_str = _MAP[raw_quality]
        elif fr.pupil_center is None and fr.limbus_center is None:
            q_str = "NO_DETECTION"
        else:
            q_str = "INSUFFICIENT"
    else:
        q_str = "NO_DETECTION"
    try:
        overall_q = DetectionQuality(q_str)
    except (ValueError, KeyError):
        overall_q = SimpleNamespace(value=q_str)

    pupil_fit_type = getattr(fr, "pupil_fit_type", None)
    limbus_fit_type = getattr(fr, "limbus_fit_type", None)

    def _make_ellipse(center, axes, angle, fit_type=None):
        if center is None or axes is None:
            return None
        full_a, full_b = float(max(axes)), float(min(axes))
        semi_a, semi_b = full_a / 2.0, full_b / 2.0
        mean_radius = (semi_a + semi_b) / 2.0
        ecc = (
            math.sqrt(max(0.0, 1.0 - (semi_b / semi_a) ** 2)) if semi_a > 0 else 0.0
        )
        circ = (semi_b / semi_a) if semi_a > 0 else 1.0
        return SimpleNamespace(
            center_x=center[0],
            center_y=center[1],
            radius=mean_radius,
            semi_major=semi_a,
            semi_minor=semi_b,
            angle_deg=angle,
            eccentricity=ecc,
            circularity=circ,
            fit_quality=fr.confidence,
            fit_rms_residual=0.0,
            num_contour_points=0,
            uncertainty_center_x=1.0,
            uncertainty_center_y=1.0,
            fit_type=fit_type,
        )

    p_ell = _make_ellipse(
        fr.pupil_center, fr.pupil_axes, fr.pupil_angle, pupil_fit_type
    )
    pupil = SimpleNamespace(
        detected=p_ell is not None,
        ellipse=p_ell,
        confidence=fr.confidence if p_ell else 0.0,
        quality=overall_q,
        method=SimpleNamespace(value="ML_optimised"),
        fit_type=pupil_fit_type,
    )
    l_ell = _make_ellipse(
        fr.limbus_center, fr.limbus_axes, fr.limbus_angle, limbus_fit_type
    )
    limbus = SimpleNamespace(
        detected=l_ell is not None,
        ellipse=l_ell,
        confidence=(min(0.95, fr.confidence + 0.05) if l_ell else 0.0),
        quality=overall_q,
        method=SimpleNamespace(value="ML_optimised"),
        fit_type=limbus_fit_type,
    )

    ref_source = getattr(fr, "corneal_reference_source", "limbus")
    has_both = pupil.detected and limbus.detected
    if has_both:
        pe, le = pupil.ellipse, limbus.ellipse
        ring_center = getattr(fr, "ring_center", None)
        use_ring_reference = (
            getattr(fr, "ring_status", "unknown") == "ring_present"
            and ring_center is not None
        )
        pts = [(pe.center_x, pe.center_y, "pupil")]
        weights = [max(pupil.confidence, 1e-3)]
        pts.append((le.center_x, le.center_y, "limbus"))
        weights.append(max(limbus.confidence, 1e-3))
        if use_ring_reference:
            pts.append((ring_center[0], ring_center[1], "ring"))
            weights.append(max(getattr(fr, "confidence", 0.0), 1e-3))
        total_w = sum(weights)
        ref_x = sum(pt[0] * w for pt, w in zip(pts, weights)) / total_w
        ref_y = sum(pt[1] * w for pt, w in zip(pts, weights)) / total_w
        ref_source = "+".join(name for _, _, name in pts)
        dx = pe.center_x - ref_x
        dy = pe.center_y - ref_y
        mag_px = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(dy, dx))
        if cal.calibrated:
            dx_mm, dy_mm = dx * cal.mm_per_px, dy * cal.mm_per_px
            mag_mm = mag_px * cal.mm_per_px
            off_mm = (dx_mm, dy_mm)
        else:
            mag_mm, off_mm = None, None
        cc = SimpleNamespace(
            valid=True,
            center_px=(ref_x, ref_y),
            offset_px=(dx, dy),
            offset_magnitude_px=mag_px,
            offset_magnitude_mm=mag_mm,
            offset_mm=off_mm,
            offset_angle_deg=ang,
        )
    else:
        cc = SimpleNamespace(
            valid=False,
            center_px=(0.0, 0.0),
            offset_px=(0.0, 0.0),
            offset_magnitude_px=0.0,
            offset_magnitude_mm=None,
            offset_mm=None,
            offset_angle_deg=0.0,
        )

    meta = SimpleNamespace(
        processing_time_ms=fr.processing_ms,
        latency_ms=getattr(fr, "latency_ms", fr.processing_ms),
        frame_number=fr.frame_number,
        image_width=W,
        image_height=H,
        source="camera (optimised)",
        reuse_cached_result=bool(getattr(fr, "reuse_cached_result", False)),
        reuse_reason=getattr(fr, "reuse_reason", None),
    )

    alerts: List[str] = []
    if fr.is_interpolated:
        alerts.append("\u26a1 Interpolated frame (Kalman prediction)")
    if fr.quality is not None and fr.quality.value == "FAILED":
        alerts.append("\u26a0 Detection failed this frame")

    result = SimpleNamespace(
        pupil=pupil,
        limbus=limbus,
        corneal_center=cc,
        calibration=cal,
        metadata=meta,
        overall_quality=overall_q,
        overall_confidence=fr.confidence,
        has_both=has_both,
        alerts=alerts,
        ring_status=getattr(fr, "ring_status", "unknown"),
        ring_center=getattr(fr, "ring_center", None),
        ring_radius=getattr(fr, "ring_radius", None),
        ring_dot_count=getattr(fr, "ring_dot_count", 0),
        corneal_reference_source=ref_source,
    )

    if frame_result_to_export_dict_fn is not None:
        result.to_dict = lambda _r=result, _fr=fr, _cal=cal, _fn=frame_result_to_export_dict_fn: (
            _fn(_fr, _cal, _r)
        )

    return result


# ── Adapted result → export dict ───────────────────────────────────


def frame_result_to_export_dict(
    fr: Any,
    cal: SimpleNamespace,
    adapted: SimpleNamespace,
) -> Dict[str, Any]:
    """Convert adapted result back to export-friendly dict for to_dict()."""
    d: Dict[str, Any] = {
        "metadata": {
            "frame_number": fr.frame_number,
            "processing_time_ms": fr.processing_ms,
            "latency_ms": getattr(fr, "latency_ms", fr.processing_ms),
            "source": "camera (optimised)",
        },
        "overall_quality": (
            adapted.overall_quality.value
            if hasattr(adapted.overall_quality, "value")
            else str(adapted.overall_quality)
        ),
        "overall_confidence": fr.confidence,
        "calibration": {
            "calibrated": cal.calibrated,
            "mm_per_px": cal.mm_per_px,
            "px_per_mm": cal.px_per_mm,
        },
    }
    if fr.pupil_center is not None and fr.pupil_axes is not None:
        semi_a, semi_b = max(fr.pupil_axes) / 2.0, min(fr.pupil_axes) / 2.0
        mean_r = (semi_a + semi_b) / 2.0
        mm = cal.mm_per_px if cal.calibrated else 0.0
        d["pupil"] = {
            "detected": True,
            "confidence": fr.confidence,
            "fit_type": getattr(fr, "pupil_fit_type", None),
            "radius_mm": (mean_r * mm) if cal.calibrated else None,
            "center_mm": (
                (fr.pupil_center[0] * mm, fr.pupil_center[1] * mm)
                if cal.calibrated else None
            ),
            "ellipse": {
                "center_x": fr.pupil_center[0],
                "center_y": fr.pupil_center[1],
                "radius": mean_r,
                "semi_major": semi_a,
                "semi_minor": semi_b,
                "angle_deg": float(getattr(fr, "pupil_angle", 0.0) or 0.0),
                "diameter_mm": (mean_r * 2.0 * mm) if cal.calibrated else None,
                "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
            },
        }
    else:
        d["pupil"] = {"detected": False, "ellipse": {}}

    if fr.limbus_center is not None and fr.limbus_axes is not None:
        semi_a, semi_b = max(fr.limbus_axes) / 2.0, min(fr.limbus_axes) / 2.0
        mean_r = (semi_a + semi_b) / 2.0
        mm = cal.mm_per_px if cal.calibrated else 0.0
        d["limbus"] = {
            "detected": True,
            "confidence": min(0.95, fr.confidence + 0.05),
            "fit_type": getattr(fr, "limbus_fit_type", None),
            "radius_mm": (mean_r * mm) if cal.calibrated else None,
            "center_mm": (
                (fr.limbus_center[0] * mm, fr.limbus_center[1] * mm)
                if cal.calibrated else None
            ),
            "ellipse": {
                "center_x": fr.limbus_center[0],
                "center_y": fr.limbus_center[1],
                "radius": mean_r,
                "semi_major": semi_a,
                "semi_minor": semi_b,
                "angle_deg": float(getattr(fr, "limbus_angle", 0.0) or 0.0),
                "diameter_mm": (mean_r * 2.0 * mm) if cal.calibrated else None,
                "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
            },
        }
    else:
        d["limbus"] = {"detected": False, "ellipse": {}}

    ring_center = getattr(fr, "ring_center", None)
    use_ring_reference = (
        getattr(fr, "ring_status", "unknown") == "ring_present"
        and ring_center is not None
    )
    if fr.pupil_center is not None and (
        use_ring_reference or fr.limbus_center is not None
    ):
        from pupil_tracking.core.corneal_center import blend_corneal_center_from_points

        blended = blend_corneal_center_from_points(
            pupil_center=fr.pupil_center,
            pupil_confidence=fr.confidence,
            limbus_center=fr.limbus_center,
            limbus_confidence=max(min(0.95, fr.confidence + 0.05), 1e-3),
            ring_center=ring_center if use_ring_reference else None,
            ring_confidence=fr.confidence if use_ring_reference else 0.0,
        )
        d["corneal_center"] = {
            "center_px": blended["center_px"],
            "offset_magnitude_px": blended["offset_magnitude_px"],
            "offset_magnitude_mm": (
                blended["offset_magnitude_px"] * cal.mm_per_px if cal.calibrated else None
            ),
            "offset_angle_deg": blended["offset_angle_deg"],
        }
        d["corneal_reference_source"] = blended["corneal_reference_source"]
    else:
        d["corneal_center"] = {}

    d["ring_status"] = getattr(fr, "ring_status", "unknown")
    if not use_ring_reference:
        d["corneal_reference_source"] = getattr(fr, "corneal_reference_source", "limbus")
    if ring_center is not None:
        d["ring_center_x"] = ring_center[0]
        d["ring_center_y"] = ring_center[1]
    d["ring_radius"] = getattr(fr, "ring_radius", None)
    d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)
    return d

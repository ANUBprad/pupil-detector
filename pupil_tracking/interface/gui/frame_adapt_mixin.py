from __future__ import annotations

import csv
import json
import math
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    DetectionQuality,
    CalibrationInfo,
    assign_quality_grade,
)
from pupil_tracking.utils.config import get_config, set_config
from pupil_tracking.utils.logger import get_logger
from pupil_tracking.utils.runtime_profile import (
    apply_runtime_optimizations,
    detect_runtime_profile,
)
from pupil_tracking.interface.theme import DarkTheme, Colors
from pupil_tracking.preprocessing.grayscale_handler import (
    GrayscaleMode,
    GrayscaleInfo,
)
from pupil_tracking.interface.frame_recorder import FrameRecorder

from pupil_tracking.interface.gui.constants import (
    _CORNEAL_DIAMETER_MM,
    _CIRCLE_DRAW_THRESHOLD,
    _QUALITY_COLORS,
    _GRAYSCALE_LABELS,
    _GRAYSCALE_COLORS,
    _GRAYSCALE_CYCLE,
    _WINDOW_TITLE,
    _MIN_WIDTH,
    _MIN_HEIGHT,
    _DISPLAY_FPS_CAP,
    _FAST_PIPELINE_AVAILABLE,
    FastInference,
    OptimizedVideoProcessor,
    AsyncCapture,
    FrameResult,
    TrackingQuality,
)


class FrameAdaptMixin:
    @staticmethod
    def _dict_to_frame_ns(d: dict) -> SimpleNamespace:
        """Convert a flat detection dict (from OptimizedVideoProcessor /
        FastInference / result_to_dict) into the SimpleNamespace that
        ``_adapt_frame_result`` expects."""
        pupil_det = d.get("pupil_detected", False)
        limbus_det = d.get("limbus_detected", False)

        if pupil_det:
            px, py = d.get("pupil_x", 0.0), d.get("pupil_y", 0.0)
            pr = d.get("pupil_radius", 0.0)
            # pupil_major/minor are already full-axis diameters from FastInference
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
            # limbus_major/minor are already full-axis diameters from FastInference
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

    def _adapt_frame_result(
        self, fr: Any, frame_shape: Tuple[int, ...]
    ) -> SimpleNamespace:
        H, W = frame_shape[:2]
        eye_result = getattr(fr, "_eye_result", None)
        if eye_result is not None:
            eye_result.metadata.image_width = W
            eye_result.metadata.image_height = H
            eye_result.metadata.frame_number = getattr(fr, "frame_number", 0)
            eye_result.metadata.latency_ms = getattr(fr, "latency_ms", fr.processing_ms)
            # ── Override detector-internal calibration with GUI mode ──
            # The internal UnifiedDetector's StabilizedCalibrator is stuck
            # on ANATOMICAL_ANCHOR (never receives set_calibration_mode).
            # Build the correct calibration from the GUI dropdown, then
            # re-compute pre-computed mm attributes so to_dict() / CSV
            # export reflect the user's selection.
            _cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
            _corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
            _fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 44.5)
            _ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)
            if _cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
                _px = max(0.1, _fixed_scale)
                new_cal = CalibrationInfo(
                    calibrated=True,
                    px_per_mm=_px,
                    mm_per_px=1.0 / _px,
                    source="fixed_manual",
                    method="fixed_manual",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=1.0,
                    corneal_diameter_assumed_mm=None,
                )
            elif _cal_mode == "RING_REFLECTION":
                _ring_r = getattr(fr, "ring_radius", None)
                if _ring_r is not None and _ring_r > 10:
                    _dia = _ring_r * 2.0
                    _px = _dia / _ring_ref_mm
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px,
                        source=f"ring_reflection_{_ring_ref_mm:.1f}mm",
                        method="ring_reflection",
                        reference_diameter_mm=_ring_ref_mm,
                        reference_diameter_px=_dia,
                        confidence=0.95,
                        corneal_diameter_assumed_mm=None,
                    )
                elif (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (fallback)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=0.85,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="ring_reflection",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=None,
                    )
            else:
                # ANATOMICAL_ANCHOR
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (optimised)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=min(0.95, getattr(eye_result, "overall_confidence", 0.0) + 0.05),
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="anatomical",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
            eye_result.calibration = new_cal
            # Clear stale pre-computed mm attributes set by
            # _add_mm_values / evaluate_clinical_wtw during the
            # original detection with the wrong calibration.
            for target in (
                getattr(eye_result, "limbus", None),
                getattr(eye_result, "pupil", None),
            ):
                if target is None:
                    continue
                for attr in (
                    "wtw_horizontal_mm",
                    "wtw_vertical_mm",
                    "wtw_mean_mm",
                    "wtw_astigmatism_mm",
                    "is_wtw_measured",
                    "wtw_validity_status",
                    "radius_mm",
                    "center_mm",
                ):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass
            # Re-compute mm values with the correct calibration
            if new_cal.calibrated:
                if (
                    getattr(eye_result, "pupil", None) is not None
                    and eye_result.pupil.detected
                    and eye_result.pupil.ellipse is not None
                ):
                    pe = eye_result.pupil.ellipse
                    eye_result.pupil.radius_mm = pe.radius * new_cal.mm_per_px
                    eye_result.pupil.center_mm = (
                        pe.center_x * new_cal.mm_per_px,
                        pe.center_y * new_cal.mm_per_px,
                    )
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and eye_result.limbus.detected
                    and eye_result.limbus.ellipse is not None
                ):
                    le = eye_result.limbus.ellipse
                    eye_result.limbus.radius_mm = le.radius * new_cal.mm_per_px
                    eye_result.limbus.center_mm = (
                        le.center_x * new_cal.mm_per_px,
                        le.center_y * new_cal.mm_per_px,
                    )
                    from pupil_tracking.calibration.spatial_calibration import (
                        evaluate_clinical_wtw,
                    )
                    h, v, m, astig, is_m, status = evaluate_clinical_wtw(
                        eye_result.limbus, new_cal,
                    )
                    eye_result.limbus.wtw_horizontal_mm = h
                    eye_result.limbus.wtw_vertical_mm = v
                    eye_result.limbus.wtw_mean_mm = m
                    eye_result.limbus.wtw_astigmatism_mm = astig
                    eye_result.limbus.is_wtw_measured = is_m
                    eye_result.limbus.wtw_validity_status = status
            return eye_result

        cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
        corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
        fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 44.5)
        ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)

        if cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = max(0.1, fixed_scale)
            cal = SimpleNamespace(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                reference_diameter_mm=0.0,
                reference_diameter_px=0.0,
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )
        elif cal_mode == "RING_REFLECTION":
            ring_radius = getattr(fr, "ring_radius", None)
            if ring_radius is not None and ring_radius > 10:
                dia_px = ring_radius * 2.0
                px_per_mm = dia_px / ring_ref_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm,
                    source=f"ring_reflection_{ring_ref_mm:.1f}mm",
                    method="ring_reflection",
                    reference_diameter_mm=ring_ref_mm,
                    reference_diameter_px=dia_px,
                    confidence=0.95,
                    corneal_diameter_assumed_mm=None,
                )
            elif fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm if px_per_mm > 0 else 0.0,
                    source="limbus_semi_major (fallback)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.85, fr.confidence),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="ring_reflection",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=None,
                )
        else:  # ANATOMICAL_ANCHOR
            if fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 0.0
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=mm_per_px,
                    source="limbus_semi_major (optimised)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.95, fr.confidence + 0.05),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="anatomical",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=corneal_mm,
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
            mean_radius = (semi_a + semi_b) / 2.0  # match EllipseParams convention
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
            alerts.append("⚡ Interpolated frame (Kalman prediction)")
        if fr.quality is not None and fr.quality.value == "FAILED":
            alerts.append("⚠ Detection failed this frame")

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
        result.to_dict = lambda _r=result, _fr=fr, _cal=cal: (
            self._frame_result_to_export_dict(_fr, _cal, _r)
        )
        return result

    def _frame_result_to_export_dict(
        self,
        fr: Any,
        cal: SimpleNamespace,
        adapted: SimpleNamespace,
    ) -> Dict[str, Any]:
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
                "source": getattr(cal, "source", "none"),
                "method": getattr(cal, "method", "anatomical"),
                "corneal_diameter_assumed_mm": getattr(cal, "corneal_diameter_assumed_mm", None),
            },
        }
        if fr.pupil_center is not None and fr.pupil_axes is not None:
            semi_a, semi_b = max(fr.pupil_axes) / 2.0, min(fr.pupil_axes) / 2.0
            # Mean radius — matches EllipseParams.radius and the Measurements
            # panel (which shows diameter = e.radius * 2). Using the mean (not
            # semi-major) is what makes the exported mm value vary per frame
            # instead of collapsing to the calibration constant.
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
            wtw_h = (2.0 * semi_a * mm) if cal.calibrated else None
            wtw_v = (2.0 * semi_b * mm) if cal.calibrated else None
            wtw_m = (mean_r * 2.0 * mm) if cal.calibrated else None
            wtw_astig = (abs(wtw_h - wtw_v)) if (wtw_h is not None and wtw_v is not None) else None
            is_wtw_m = bool(cal.calibrated and getattr(cal, "method", "anatomical") != "anatomical")
            if not cal.calibrated:
                wtw_status = "UNAVAILABLE"
            elif not is_wtw_m:
                wtw_status = "ANCHORED_BASELINE"
            else:
                wtw_status = "VALID_CLINICAL_RANGE" if (wtw_m is not None and 9.5 <= wtw_m <= 13.5) else "OUT_OF_BOUNDS_WARNING"

            d["limbus"] = {
                "detected": True,
                "confidence": min(0.95, fr.confidence + 0.05),
                "fit_type": getattr(fr, "limbus_fit_type", None),
                "radius_mm": (mean_r * mm) if cal.calibrated else None,
                "center_mm": (
                    (fr.limbus_center[0] * mm, fr.limbus_center[1] * mm)
                    if cal.calibrated else None
                ),
                "wtw_horizontal_mm": wtw_h,
                "wtw_vertical_mm": wtw_v,
                "wtw_mean_mm": wtw_m,
                "wtw_astigmatism_mm": wtw_astig,
                "is_wtw_measured": is_wtw_m,
                "wtw_validity_status": wtw_status,
                "ellipse": {
                    "center_x": fr.limbus_center[0],
                    "center_y": fr.limbus_center[1],
                    "radius": mean_r,
                    "semi_major": semi_a,
                    "semi_minor": semi_b,
                    "angle_deg": float(getattr(fr, "limbus_angle", 0.0) or 0.0),
                    # In anatomical mode, mean_r*2*mm ≡ wtw_m (algebraically
                    # identical).  Use wtw_m explicitly so the CSV semantically
                    # agrees with the WTW card rather than re-deriving from px.
                    "diameter_mm": wtw_m if (cal.calibrated and wtw_m is not None) else ((mean_r * 2.0 * mm) if cal.calibrated else None),
                    "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                    "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
                },
            }
        else:
            d["limbus"] = {
                "detected": False,
                "ellipse": {},
                "is_wtw_measured": False,
                "wtw_validity_status": "UNAVAILABLE",
            }

        ring_center = getattr(fr, "ring_center", None)
        use_ring_reference = (
            getattr(fr, "ring_status", "unknown") == "ring_present"
            and ring_center is not None
        )
        if fr.pupil_center is not None and (
            use_ring_reference or fr.limbus_center is not None
        ):
            points = [(fr.pupil_center[0], fr.pupil_center[1], "pupil")]
            weights = [max(fr.confidence, 1e-3)]
            if fr.limbus_center is not None:
                points.append((fr.limbus_center[0], fr.limbus_center[1], "limbus"))
                weights.append(max(min(0.95, fr.confidence + 0.05), 1e-3))
            if use_ring_reference:
                points.append((ring_center[0], ring_center[1], "ring"))
                weights.append(max(fr.confidence, 1e-3))
            total_w = sum(weights)
            ref_center = (
                sum(pt[0] * w for pt, w in zip(points, weights)) / total_w,
                sum(pt[1] * w for pt, w in zip(points, weights)) / total_w,
            )
            dx = fr.pupil_center[0] - ref_center[0]
            dy = fr.pupil_center[1] - ref_center[1]
            mag_px = math.hypot(dx, dy)
            d["corneal_center"] = {
                "center_px": ref_center,
                "offset_magnitude_px": mag_px,
                "offset_magnitude_mm": (
                    mag_px * cal.mm_per_px if cal.calibrated else None
                ),
                "offset_angle_deg": math.degrees(math.atan2(dy, dx)),
            }
        else:
            d["corneal_center"] = {}
        d["ring_status"] = getattr(fr, "ring_status", "unknown")
        d["corneal_reference_source"] = getattr(fr, "corneal_reference_source", "limbus")
        if ring_center is not None:
            d["ring_center_x"] = ring_center[0]
            d["ring_center_y"] = ring_center[1]
        d["ring_radius"] = getattr(fr, "ring_radius", None)
        d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)
        return d

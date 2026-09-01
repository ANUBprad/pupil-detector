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


class PanelsMixin:
    def _update_measurements(self, result: Any) -> None:
        try:
            cal = result.calibration
            has_cal = cal.calibrated if cal else False
            mm_per_px = cal.mm_per_px if has_cal else 0.0
            quality = (
                result.overall_quality.value
                if hasattr(result.overall_quality, "value")
                else str(result.overall_quality)
            )
            color = _QUALITY_COLORS.get(quality, "#888888")
            self._quality_label.config(
                text=f"  {quality} ({result.overall_confidence:.3f})  ",
                foreground=color,
            )
            self._summary_quality_var.set(quality)
            self._summary_quality_label.config(foreground=color)

            if result.pupil.detected and result.pupil.ellipse is not None:
                e = result.pupil.ellipse
                dia_px = e.radius * 2.0
                self._pv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._pv["diameter_px"].set(f"{dia_px:.1f} px")
                self._pv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._pv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._pv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._pv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                self._pv["fit_type"].set(ft if ft else "—")
                self._pv["confidence"].set(f"{result.pupil.confidence:.3f}")
                q_val = (
                    result.pupil.quality.value
                    if hasattr(result.pupil.quality, "value")
                    else str(result.pupil.quality)
                )
                self._pv["quality"].set(q_val)
            else:
                for var in self._pv.values():
                    var.set("---")

            if result.limbus.detected and result.limbus.ellipse is not None:
                e = result.limbus.ellipse
                dia_px = e.radius * 2.0
                self._lv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._lv["diameter_px"].set(f"{dia_px:.1f} px")
                self._lv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._lv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._lv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._lv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                self._lv["fit_type"].set(ft if ft else "—")
                self._lv["confidence"].set(f"{result.limbus.confidence:.3f}")
                q_val = (
                    result.limbus.quality.value
                    if hasattr(result.limbus.quality, "value")
                    else str(result.limbus.quality)
                )
                self._lv["quality"].set(q_val)
            else:
                for var in self._lv.values():
                    var.set("---")

            cc = result.corneal_center
            if cc.valid and result.has_both:
                pe = result.pupil.ellipse
                le = result.limbus.ellipse
                self._ov["corneal_center"].set(
                    f"({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f}) px"
                )
                self._ov["corneal_reference"].set(
                    getattr(result, "corneal_reference_source", "limbus")
                )
                ring_center = getattr(result, "ring_center", None)
                if ring_center is not None:
                    self._ov["ring_center"].set(
                        f"({ring_center[0]:.1f}, {ring_center[1]:.1f}) px"
                    )
                else:
                    self._ov["ring_center"].set("---")
                dx, dy = cc.offset_px
                offset_px = cc.offset_magnitude_px
                offset_angle = cc.offset_angle_deg
                self._ov["offset_px"].set(f"{offset_px:.1f} px")
                self._ov["offset_vec_px"].set(f"({dx:.1f}, {dy:.1f}) px")
                if has_cal:
                    dx_mm, dy_mm = dx * mm_per_px, dy * mm_per_px
                    offset_mm = offset_px * mm_per_px
                    self._ov["offset_mm"].set(f"{offset_mm:.3f} mm")
                    self._ov["offset_vec_mm"].set(f"({dx_mm:.3f}, {dy_mm:.3f}) mm")
                else:
                    self._ov["offset_mm"].set("— (no calibration)")
                    self._ov["offset_vec_mm"].set("— (no calibration)")
                self._ov["offset_angle"].set(f"{offset_angle:.1f}°")
                ring_radius = getattr(result, "ring_radius", None)
                if ring_radius is not None:
                    ring_dia_px = ring_radius * 2.0
                    self._ov["ring_diameter_px"].set(f"{ring_dia_px:.1f} px")
                    if has_cal:
                        self._ov["ring_diameter_mm"].set(
                            f"{ring_dia_px * mm_per_px:.3f} mm"
                        )
                    else:
                        self._ov["ring_diameter_mm"].set("â€” (no calibration)")
                else:
                    self._ov["ring_diameter_px"].set("---")
                    self._ov["ring_diameter_mm"].set("---")
                if le.radius > 0:
                    self._ov["pupil_limbus_ratio"].set(f"{pe.radius / le.radius:.3f}")
                else:
                    self._ov["pupil_limbus_ratio"].set("---")
            else:
                for var in self._ov.values():
                    var.set("---")

            if has_cal:
                method_tag = getattr(cal, "method", "anatomical")
                self._cv_vars["source"].set(f"{cal.source} [{method_tag}]")
                self._cv_vars["scale_px"].set(f"{cal.px_per_mm:.2f} px/mm")
                self._cv_vars["scale_mm"].set(f"{cal.mm_per_px:.4f} mm/px")
                if getattr(cal, "corneal_diameter_assumed_mm", None):
                    self._cv_vars["reference"].set(
                        f"Assumed {cal.corneal_diameter_assumed_mm:.1f}mm"
                    )
                elif cal.reference_diameter_mm > 0:
                    self._cv_vars["reference"].set(
                        f"{cal.reference_diameter_mm:.1f}mm ({cal.reference_diameter_px:.0f}px)"
                    )
                else:
                    self._cv_vars["reference"].set("Fixed External Scale")
            else:
                self._cv_vars["source"].set("not calibrated")
                self._cv_vars["scale_px"].set("---")
                self._cv_vars["scale_mm"].set("---")
                self._cv_vars["reference"].set("---")

            # ── Cyclotorsion / Iris Card ──
            iris_det = getattr(result, "iris_detection", None)
            if (
                hasattr(self, "_iris_vars")
                and iris_det is not None
                and getattr(iris_det, "valid", False)
            ):
                fs = getattr(iris_det, "feature_set", None)
                n_features = len(fs.features) if fs is not None else 0
                coverage = getattr(fs, "region_coverage", 0.0) if fs is not None else 0.0
                self._iris_vars["status"].set("Valid")
                self._iris_vars["feature_count"].set(str(n_features))
                self._iris_vars["angular_coverage"].set(f"{coverage:.1%}")

                # Cyclotorsion (if paired result available)
                corr = getattr(result, "iris_correspondence", None)
                if corr is not None and getattr(corr, "valid", False):
                    rot = corr.estimated_rotation_deg
                    sign = "+" if rot >= 0 else ""
                    self._iris_vars["rotation_angle"].set(f"{sign}{rot:.2f}°")
                    self._iris_vars["confidence"].set("High")
                    self._iris_vars["evidence"].set("Good")
                else:
                    self._iris_vars["rotation_angle"].set("---")
                    self._iris_vars["confidence"].set("---")
                    self._iris_vars["evidence"].set("Single image")
            elif hasattr(self, "_iris_vars"):
                # Check why iris is unavailable
                iris_status = getattr(result, "iris_status", None)
                if iris_status is not None:
                    status_str = (
                        iris_status.value
                        if hasattr(iris_status, "value")
                        else str(iris_status)
                    )
                    self._iris_vars["status"].set(f"Rejected: {status_str}")
                else:
                    self._iris_vars["status"].set("Unavailable")
                self._iris_vars["feature_count"].set("---")
                self._iris_vars["angular_coverage"].set("---")
                self._iris_vars["rotation_angle"].set("---")
                self._iris_vars["confidence"].set("---")
                self._iris_vars["evidence"].set("---")

            # ── Corneal Dimensions (WTW) Card ──
            limbus_res = getattr(result, "limbus", None)
            if (
                hasattr(self, "_wtw_vars")
                and limbus_res is not None
                and getattr(limbus_res, "detected", False)
                and getattr(limbus_res, "ellipse", None) is not None
                and has_cal
            ):
                le = limbus_res.ellipse
                # Always recompute WTW from current pixel geometry and
                # calibration.  Pre-computed attributes (set by
                # _add_mm_values during detection) become stale when the
                # calibration mode changes without a new detection.
                h_wtw = 2.0 * le.semi_major * mm_per_px
                v_wtw = 2.0 * le.semi_minor * mm_per_px
                m_wtw = (h_wtw + v_wtw) / 2.0

                self._wtw_vars["horizontal"].set(f"{h_wtw:.2f} mm")
                self._wtw_vars["vertical"].set(f"{v_wtw:.2f} mm")
                self._wtw_vars["mean"].set(f"{m_wtw:.2f} mm")
            elif hasattr(self, "_wtw_vars"):
                for var in self._wtw_vars.values():
                    var.set("---")


            proc_ms = float(getattr(result.metadata, "processing_time_ms", 0.0) or 0.0)
            reused = bool(getattr(result.metadata, "reuse_cached_result", False))
            if not reused and proc_ms > 0.5:
                self._last_real_proc_time_ms = proc_ms
            shown_proc_ms = getattr(self, "_last_real_proc_time_ms", proc_ms)
            if not reused:
                shown_proc_ms = proc_ms
            display_proc_prev = getattr(self, "_display_proc_time_ms", shown_proc_ms)
            proc_alpha = 0.18 if reused else 0.32
            display_proc_ms = display_proc_prev + proc_alpha * (shown_proc_ms - display_proc_prev)
            self._display_proc_time_ms = display_proc_ms
            self._proc_time_var.set(f"{display_proc_ms:.1f} ms")
            latency_ms = float(getattr(
                result.metadata, "latency_ms", result.metadata.processing_time_ms
            ) or 0.0)
            display_latency_prev = getattr(self, "_display_latency_ms", latency_ms)
            latency_alpha = 0.20 if reused else 0.34
            display_latency_ms = (
                display_latency_prev + latency_alpha * (latency_ms - display_latency_prev)
            )
            self._display_latency_ms = display_latency_ms
            self._latency_var.set(f"{display_latency_ms:.1f} ms")
            if not self._using_optimized_camera:
                self._latency_avg_var.set("---")
                self._drop_var.set("---")
                self._tracking_state_var.set("---")
            self._frame_var.set(str(result.metadata.frame_number))
            self._image_size_var.set(
                f"{result.metadata.image_width} × {result.metadata.image_height}"
            )
            self._summary_latency_var.set(f"{display_latency_ms:.1f} ms")
            self._summary_pipeline_var.set(self._pipeline_var.get())
            tracking_text = self._tracking_state_var.get()
            if not tracking_text or tracking_text == "---":
                tracking_text = "Ready" if result.has_both else "Waiting"
            self._set_summary_tracking_state(tracking_text)

            mode = self._grayscale_mode_var.get()
            mode_labels = {
                "off": "RGB (Original)",
                "auto": "Auto-Detect",
                "force": "Forced Grayscale",
            }
            gs_label = mode_labels.get(mode, mode)
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gs_label += " ✓ applied"
            self._gray_mode_var_display.set(gs_label)

            if hasattr(self, "_gray_settings_status"):
                self._gray_settings_status.set(f"Current: {gs_label}")

            self._update_details(result)
        except Exception as exc:
            self.logger.exception("Measurement update error: %s", exc)
            self._report_runtime_issue(
                "Measurement panel recovered from an internal error"
            )

    def _update_details(self, result: Any) -> None:
        cal = result.calibration
        has_cal = cal.calibrated if cal else False
        mm_per_px = cal.mm_per_px if has_cal else 0.0

        self._details_text.config(state=tk.NORMAL)
        self._details_text.delete("1.0", tk.END)

        lines: List[str] = []
        lines.append(f"Source:  {result.metadata.source}")
        lines.append(
            f"Image:   {result.metadata.image_width}×{result.metadata.image_height}"
        )
        q_val = (
            result.overall_quality.value
            if hasattr(result.overall_quality, "value")
            else str(result.overall_quality)
        )
        lines.append(f"Quality: {q_val} ({result.overall_confidence:.4f})")
        lines.append(f"Time:    {result.metadata.processing_time_ms:.1f} ms")
        latency_ms = getattr(result.metadata, "latency_ms", None)
        if latency_ms is not None:
            lines.append(f"Latency: {latency_ms:.1f} ms")

        # Grayscale info in details
        mode = self._grayscale_mode_var.get()
        mode_names = {"off": "OFF (RGB)", "auto": "AUTO", "force": "FORCE (Grayscale)"}
        lines.append(f"Gray:    {mode_names.get(mode, mode)}")
        if self._detector is not None:
            gs_info = self._detector.last_grayscale_info
            if gs_info is not None:
                lines.append(
                    f"         applied={gs_info.conversion_applied}, "
                    f"input={'gray' if gs_info.was_grayscale else 'RGB'}"
                )
                if gs_info.conversion_applied:
                    lines.append(
                        f"         contrast {gs_info.contrast_before:.1f} "
                        f"→ {gs_info.contrast_after:.1f}"
                    )
        lines.append("")

        ring_status = getattr(result, "ring_status", "unknown")
        ring_center = getattr(result, "ring_center", None)
        ring_radius = getattr(result, "ring_radius", None)
        if ring_status == "ring_present" and ring_center is not None and ring_radius is not None:
            lines.append("=== SUCTION RING ===")
            lines.append(f"  Status:     {ring_status}")
            lines.append(f"  Method:     {getattr(result, 'ring_method', 'unknown')}")
            lines.append(
                f"  Center:     ({ring_center[0]:.2f}, {ring_center[1]:.2f}) px"
            )
            lines.append(f"  Diameter:   {ring_radius * 2.0:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {ring_radius * 2.0 * mm_per_px:.3f} mm")
            lines.append(f"  Dots:       {getattr(result, 'ring_dot_count', 0)}")
            lines.append(
                f"  Reference:  {getattr(result, 'corneal_reference_source', 'limbus')}"
            )
            lines.append("")

        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.pupil.method.value
                if hasattr(result.pupil.method, "value")
                else str(result.pupil.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(result.pupil, "fit_type", None)
            lines.append("=== PUPIL ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append(
                f"  Uncert:     ±({e.uncertainty_center_x:.2f},"
                f" {e.uncertainty_center_y:.2f}) px"
            )
            lines.append("")

        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.limbus.method.value
                if hasattr(result.limbus.method, "value")
                else str(result.limbus.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(
                result.limbus, "fit_type", None
            )
            lines.append("=== LIMBUS ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append("")

        if result.corneal_center.valid:
            cc = result.corneal_center
            lines.append("=== CORNEAL CENTRE & OFFSET ===")
            lines.append(
                f"  Centre:     ({cc.center_px[0]:.2f}, {cc.center_px[1]:.2f}) px"
            )
            lines.append(
                f"  Offset:     ({cc.offset_px[0]:.2f}, {cc.offset_px[1]:.2f}) px"
            )
            lines.append(f"  Magnitude:  {cc.offset_magnitude_px:.2f} px")
            if cc.offset_magnitude_mm is not None:
                lines.append(
                    f"  Offset:     ({cc.offset_mm[0]:.3f}, {cc.offset_mm[1]:.3f}) mm"
                )
                lines.append(f"  Magnitude:  {cc.offset_magnitude_mm:.3f} mm")
            lines.append(f"  Angle:      {cc.offset_angle_deg:.2f}°")
            lines.append("")

        if has_cal:
            lines.append("=== CALIBRATION ===")
            lines.append(f"  Source:     {cal.source}")
            lines.append(f"  px/mm:      {cal.px_per_mm:.4f}")
            lines.append(f"  mm/px:      {cal.mm_per_px:.6f}")
            lines.append(
                f"  Ref diam:   {cal.reference_diameter_mm:.1f}"
                f" mm = {cal.reference_diameter_px:.0f} px"
            )
            lines.append(f"  Confidence: {cal.confidence:.3f}")
            lines.append("")

        iris_det = getattr(result, "iris_detection", None)
        if iris_det is not None and getattr(iris_det, "valid", False):
            fs = getattr(iris_det, "feature_set", None)
            n_features = len(fs.features) if fs is not None else 0
            coverage = getattr(fs, "region_coverage", 0.0) if fs is not None else 0.0
            lines.append("=== IRIS FEATURES ===")
            lines.append(f"  Features:   {n_features}")
            lines.append(f"  Coverage:   {coverage:.1%}")
            lines.append(f"  Status:     {iris_det.status.value}")
            lines.append("")
        elif getattr(result, "iris_status", None) is not None:
            iris_status = result.iris_status
            status_str = (
                iris_status.value
                if hasattr(iris_status, "value")
                else str(iris_status)
            )
            lines.append("=== IRIS FEATURES ===")
            lines.append(f"  Status:     {status_str}")
            lines.append("")

        if result.alerts:
            lines.append("=== ALERTS ===")
            for alert in result.alerts:
                lines.append(f"  ! {alert}")
            lines.append("")

        self._details_text.insert("1.0", "\n".join(lines))
        self._details_text.config(state=tk.DISABLED)

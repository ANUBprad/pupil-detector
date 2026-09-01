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


class ExportMixin:
    def _export_csv(self) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        def _round(v: Any, nd: int = 4) -> Any:
            """Round numeric values; pass through blanks/None as ''."""
            if v is None or v == "":
                return ""
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    return ""
                return round(fv, nd)
            except (TypeError, ValueError):
                return v

        def _diam_mm(ell: Dict[str, Any], mm_px: float) -> Any:
            """Prefer the pre-computed diameter_mm; else derive from mean
            radius. NEVER recompute from semi-major alone (that collapses to
            the calibration constant). This matches the Measurements panel,
            which shows diameter = ellipse.radius * 2 * mm_per_px."""
            if ell.get("diameter_mm") not in (None, ""):
                return _round(ell.get("diameter_mm"))
            r = ell.get("radius")
            if r not in (None, "") and mm_px:
                return _round(float(r) * 2.0 * mm_px)
            return ""

        rows: List[Dict[str, Any]] = []
        for r in self._results_history:
            pupil = r.get("pupil", {})
            limbus = r.get("limbus", {})
            pe = pupil.get("ellipse", {})
            le = limbus.get("ellipse", {})
            cc = r.get("corneal_center", {})
            meta = r.get("metadata", {})
            cal_info = r.get("calibration", {})
            is_cal = bool(cal_info.get("calibrated", False))
            mm_px = float(cal_info.get("mm_per_px", 0) or 0) if is_cal else 0.0
            cal_method = cal_info.get("method", "") or ("anatomical" if is_cal else "")
            assumed_mm = cal_info.get("corneal_diameter_assumed_mm", "")
            if assumed_mm is None or not is_cal or cal_method != "anatomical":
                assumed_mm = ""

            # Diameter in px = mean radius * 2 (matches the panel exactly).
            pupil_dia_px = (pe.get("radius", 0) or 0) * 2 if pe.get("radius") else ""
            limbus_dia_px = (le.get("radius", 0) or 0) * 2 if le.get("radius") else ""

            # Semi-axes in mm: dynamically computed from semi_major/minor px * mm_per_px
            pupil_major_mm = _round(float(pe.get("semi_major", 0)) * mm_px) if (is_cal and pe.get("semi_major")) else ""
            pupil_minor_mm = _round(float(pe.get("semi_minor", 0)) * mm_px) if (is_cal and pe.get("semi_minor")) else ""
            limbus_major_mm = _round(float(le.get("semi_major", 0)) * mm_px) if (is_cal and le.get("semi_major")) else ""
            limbus_minor_mm = _round(float(le.get("semi_minor", 0)) * mm_px) if (is_cal and le.get("semi_minor")) else ""

            rows.append(
                {
                    "frame": meta.get("frame_number", ""),
                    "source": meta.get("source", ""),
                    "processing_time_ms": _round(meta.get("processing_time_ms", ""), 2),
                    "latency_ms": _round(meta.get("latency_ms", ""), 2),
                    # ── Pupil ─────────────────────────────────────────────
                    "pupil_detected": pupil.get("detected", False),
                    "pupil_cx_px": _round(pe.get("center_x", "")),
                    "pupil_cy_px": _round(pe.get("center_y", "")),
                    "pupil_diameter_px": _round(pupil_dia_px),
                    "pupil_diameter_mm": _diam_mm(pe, mm_px) if is_cal else "",
                    "pupil_radius_mm": _round(pupil.get("radius_mm", "")) if is_cal else "",
                    "pupil_semi_major_px": _round(pe.get("semi_major", "")),
                    "pupil_semi_minor_px": _round(pe.get("semi_minor", "")),
                    "pupil_semi_major_mm": pupil_major_mm,
                    "pupil_semi_minor_mm": pupil_minor_mm,
                    "pupil_angle_deg": _round(pe.get("angle_deg", ""), 2),
                    "pupil_fit_type": pupil.get("fit_type", "") or "",
                    "pupil_confidence": _round(pupil.get("confidence", ""), 3),
                    # ── Limbus ────────────────────────────────────────────
                    "limbus_detected": limbus.get("detected", False),
                    "limbus_cx_px": _round(le.get("center_x", "")),
                    "limbus_cy_px": _round(le.get("center_y", "")),
                    "limbus_diameter_px": _round(limbus_dia_px),
                    "limbus_diameter_mm": _diam_mm(le, mm_px) if is_cal else "",
                    "limbus_radius_mm": _round(limbus.get("radius_mm", "")) if is_cal else "",
                    "limbus_semi_major_px": _round(le.get("semi_major", "")),
                    "limbus_semi_minor_px": _round(le.get("semi_minor", "")),
                    "limbus_semi_major_mm": limbus_major_mm,
                    "limbus_semi_minor_mm": limbus_minor_mm,
                    "limbus_angle_deg": _round(le.get("angle_deg", ""), 2),
                    "limbus_fit_type": limbus.get("fit_type", "") or "",
                    "limbus_confidence": _round(limbus.get("confidence", ""), 3),
                    # ── Clinical Corneal WTW Dimensions ───────────────────
                    "measured_wtw_horizontal_mm": _round(limbus.get("wtw_horizontal_mm", "")) if is_cal else "",
                    "measured_wtw_vertical_mm": _round(limbus.get("wtw_vertical_mm", "")) if is_cal else "",
                    "measured_wtw_mean_mm": _round(limbus.get("wtw_mean_mm", "")) if is_cal else "",
                    "limbus_astigmatic_difference_mm": _round(limbus.get("wtw_astigmatism_mm", "")) if is_cal else "",
                    "wtw_validity_status": limbus.get("wtw_validity_status", "") if is_cal else "",
                    # ── Corneal center / offset ───────────────────────────
                    "corneal_center_x_px": _round(
                        (cc.get("center_px") or ["", ""])[0]
                    ),

                    "corneal_center_y_px": _round(
                        (cc.get("center_px") or ["", ""])[1]
                    ),
                    "offset_px": _round(cc.get("offset_magnitude_px", "")),
                    "offset_mm": _round(cc.get("offset_magnitude_mm", "")) if is_cal else "",
                    "offset_angle_deg": _round(cc.get("offset_angle_deg", ""), 2),
                    "corneal_reference_source": r.get("corneal_reference_source", ""),
                    # ── Ring ──────────────────────────────────────────────
                    "ring_status": r.get("ring_status", ""),
                    "ring_center_x": _round(r.get("ring_center_x", "")),
                    "ring_center_y": _round(r.get("ring_center_y", "")),
                    "ring_radius_px": _round(r.get("ring_radius", "")),
                    "ring_diameter_mm": (
                        _round((r.get("ring_radius") or 0) * 2.0 * mm_px)
                        if is_cal and r.get("ring_radius") and mm_px else ""
                    ),
                    "ring_dot_count": r.get("ring_dot_count", ""),
                    # ── Calibration / quality ─────────────────────────────
                    "calibrated": is_cal,
                    "calibration_method": cal_method if is_cal else "",
                    "corneal_diameter_assumed_mm": _round(assumed_mm, 2) if assumed_mm != "" else "",
                    "px_per_mm": _round(cal_info.get("px_per_mm", "")) if is_cal else "",
                    "mm_per_px": _round(cal_info.get("mm_per_px", ""), 6) if is_cal else "",
                    "quality": r.get("overall_quality", ""),
                    "overall_confidence": _round(r.get("overall_confidence", ""), 3),
                    "grayscale_mode": r.get("grayscale_mode", ""),
                }
            )

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        self._status_var.set(f"Exported {len(rows)} rows → {path}")

    def _export_json(self) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        path = filedialog.asksaveasfilename(
            title="Export JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        export_payload = {
            "export_info": {
                "version": "2.3",
                "total_frames": len(self._results_history),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "corneal_diameter_assumption_mm": _CORNEAL_DIAMETER_MM,
            },
            "results": self._results_history,
        }
        with open(path, "w") as fh:
            json.dump(export_payload, fh, indent=2, default=str)
        self._status_var.set(f"Exported {len(self._results_history)} results → {path}")

    def _export_snapshot(self) -> None:
        if self._current_image is None:
            messagebox.showinfo("No Image", "No image to export")
            return
        path = filedialog.asksaveasfilename(
            title="Save Snapshot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
        )
        if not path:
            return
        image = self._prepare_recording_frame(
            self._current_image,
            self._current_result,
        )
        cv2.imwrite(path, image)
        self._status_var.set(f"Snapshot saved → {path}")

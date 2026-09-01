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


class CalibrationMixin:
    def _cancel_ruler_calibration(self, event: Any = None) -> None:
        self._ruler_calibration_active = False
        self._ruler_points.clear()
        self._canvas.configure(cursor="crosshair")
        self._status_var.set("Ruler calibration cancelled")
        self._refresh_display()

    def _start_ruler_calibration(self, known_dist_mm: float = 10.0) -> None:
        self._ruler_calibration_active = True
        self._ruler_points.clear()
        self._ruler_known_dist_mm_var.set(known_dist_mm)
        self._canvas.configure(cursor="tcross")
        self._status_var.set(
            f"Ruler Calibration: Click Point 1 on image (known span: {known_dist_mm:.1f} mm)"
        )
        self._refresh_display()

    def _handle_ruler_canvas_click(self, event: Any) -> None:
        pt = self._canvas_to_image_point(event.x, event.y)
        if pt is None:
            return
        self._ruler_points.append(pt)
        if len(self._ruler_points) == 1:
            self._status_var.set(
                f"Ruler point 1 set at ({pt[0]:.1f}, {pt[1]:.1f}) px. Now click Point 2..."
            )
            self._refresh_display()
        elif len(self._ruler_points) >= 2:
            p1 = self._ruler_points[0]
            p2 = self._ruler_points[1]
            dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            if known_mm <= 0:
                known_mm = 10.0
            if dist_px < 2.0:
                self._status_var.set("Selected points too close together (<2px). Click 2 distinct points.")
                self._ruler_points.clear()
                self._refresh_display()
                return

            from pupil_tracking.calibration.spatial_calibration import calculate_ruler_scale
            px_per_mm, mm_per_px = calculate_ruler_scale(p1, p2, known_mm)
            self._fixed_scale_var.set(round(px_per_mm, 2))
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            self._ruler_calibration_active = False
            self._canvas.configure(cursor="crosshair")
            msg = f"Ruler Calibrated: {px_per_mm:.2f} px/mm ({dist_px:.1f} px = {known_mm:.1f} mm)"
            self._status_var.set(msg)
            self._refresh_display()
            messagebox.showinfo(
                "Ruler Calibration Applied",
                f"Calibration Successful!\n\nScale: {px_per_mm:.2f} px/mm ({mm_per_px:.4f} mm/px)\n"
                f"Measured Span: {dist_px:.1f} px = {known_mm:.1f} mm\n"
                f"Mode: Fixed Pixel Scale (Active)",
            )

    def _open_calibration_wizard(self) -> None:
        """Open a modern, comprehensive Calibration & Corneal Sizing Wizard dialog."""
        wizard = tk.Toplevel(self.root)
        wizard.title("Scale & Corneal Sizing (WTW) Calibration Wizard")
        wizard.geometry("540x580")
        wizard.resizable(False, False)
        wizard.transient(self.root)
        wizard.grab_set()

        c = self._colors
        wizard.configure(bg=c.BG_PRIMARY)

        # Header
        header = ttk.Frame(wizard, padding=16)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Optical Scale & WTW Calibration",
            font=("Segoe UI", 14, "bold"),
            foreground=c.FG_PRIMARY,
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Establish an accurate pixel-to-millimeter ratio for genuine patient-specific\nWhite-to-White corneal diameter measurements.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        # Main cards container
        container = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        container.pack(fill=tk.BOTH, expand=True)

        # Method 1: Interactive 2-Point Ruler Card
        m1 = ttk.LabelFrame(container, text="Method 1: Interactive 2-Point Ruler Tool", padding=10)
        m1.pack(fill=tk.X, pady=4)
        ttk.Label(
            m1,
            text="Click two points on an image with a known physical distance\n(e.g., surgical marker, ruler, or target grid).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ruler_row = ttk.Frame(m1)
        ruler_row.pack(fill=tk.X, pady=2)
        ttk.Label(ruler_row, text="Known Distance (mm):", width=20).pack(side=tk.LEFT)
        known_mm_ent = ttk.Spinbox(
            ruler_row,
            from_=0.5,
            to=100.0,
            increment=0.5,
            textvariable=self._ruler_known_dist_mm_var,
            width=8,
        )
        known_mm_ent.pack(side=tk.LEFT, padx=4)

        def _start_ruler():
            wizard.destroy()
            self._start_ruler_calibration(self._ruler_known_dist_mm_var.get())

        ttk.Button(
            ruler_row,
            text="🎯 Click 2 Points on Canvas",
            command=_start_ruler,
        ).pack(side=tk.LEFT, padx=8)

        # Method 2: Direct Optical Scale Input
        m2 = ttk.LabelFrame(container, text="Method 2: Direct Optical / Microscope Scale", padding=10)
        m2.pack(fill=tk.X, pady=4)
        ttk.Label(
            m2,
            text="Directly enter known magnification / scale factor in pixels per millimeter.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        scale_row = ttk.Frame(m2)
        scale_row.pack(fill=tk.X, pady=2)
        ttk.Label(scale_row, text="Fixed Scale (px/mm):", width=20).pack(side=tk.LEFT)
        scale_ent = ttk.Spinbox(
            scale_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
        )
        scale_ent.pack(side=tk.LEFT, padx=4)

        def _apply_fixed():
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Scale Set", f"Fixed optical scale set to {self._fixed_scale_var.get():.2f} px/mm.")

        ttk.Button(scale_row, text="Apply Scale", command=_apply_fixed).pack(side=tk.LEFT, padx=8)

        # Method 3: Ring Fiducial Reflection
        m3 = ttk.LabelFrame(container, text="Method 3: Suction / Placido Ring Fiducial", padding=10)
        m3.pack(fill=tk.X, pady=4)
        ttk.Label(
            m3,
            text="Calibrate dynamically from a known surgical suction ring or Placido ring.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ring_row = ttk.Frame(m3)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ring Outer Dia (mm):", width=20).pack(side=tk.LEFT)
        ring_ent = ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=25.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
        )
        ring_ent.pack(side=tk.LEFT, padx=4)

        def _apply_ring():
            self._calibration_mode_var.set("RING_REFLECTION")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Ring Mode Set", f"Ring reflection calibration activated ({self._ring_ref_mm_var.get():.1f} mm standard).")

        ttk.Button(ring_row, text="Apply Ring Mode", command=_apply_ring).pack(side=tk.LEFT, padx=8)

        # Method 4: Anatomical Baseline Anchor
        m4 = ttk.LabelFrame(container, text="Method 4: Anatomical Baseline Anchor (12.0 mm)", padding=10)
        m4.pack(fill=tk.X, pady=4)
        ttk.Label(
            m4,
            text="Assumes horizontal corneal diameter is 12.0 mm (not patient-specific).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        def _apply_anatomical():
            self._calibration_mode_var.set("ANATOMICAL_ANCHOR")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Baseline Set", "Calibration set to 12.0 mm Anatomical Anchor.")

        ttk.Button(m4, text="Reset to 12.0 mm Baseline Anchor", command=_apply_anatomical).pack(anchor=tk.W)

        # Footer close button
        btn_bar = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        btn_bar.pack(fill=tk.X)
        ttk.Button(btn_bar, text="Close", command=wizard.destroy).pack(side=tk.RIGHT)

    def _sync_calibration_to_detector(self) -> None:
        """Push current GUI calibration settings into the detector immediately.

        Called synchronously before every detection so that the detector
        always reflects the user's latest calibration choice, even when
        the 180 ms debounced ``_apply_live_settings`` has not fired yet.
        """
        if self._detector is None:
            return
        cal_mode = self._calibration_mode_var.get()
        manual_px = float(self._fixed_scale_var.get())
        corneal_mm = float(self._corneal_ref_mm_var.get())
        ring_mm = float(self._ring_ref_mm_var.get())
        if hasattr(self.cfg, "calibration"):
            self.cfg.calibration.mode = cal_mode
            self.cfg.calibration.manual_px_per_mm = manual_px
            self.cfg.calibration.corneal_diameter_mm = corneal_mm
            self.cfg.calibration.suction_ring_diameter_mm = ring_mm
        self._detector.set_calibration_mode(
            mode=cal_mode,
            manual_px_per_mm=manual_px,
            corneal_diameter_mm=corneal_mm,
            ring_diameter_mm=ring_mm,
        )

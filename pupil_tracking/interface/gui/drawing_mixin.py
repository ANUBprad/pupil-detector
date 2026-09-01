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


class DrawingMixin:
    def _refresh_display(self) -> None:
        try:
            if self._current_image is None:
                self._draw_welcome_screen()
                return
            if not self._canvas.winfo_ismapped():
                return
        except Exception as exc:
            self.logger.exception("Display pre-check error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")
            return

        if self._current_image is None:
            self._draw_welcome_screen()
            return
        if not self._canvas.winfo_ismapped():
            return

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 10 of 12 — Convert display frame
        #
        # The detector already processed the original image.
        # Now convert the DISPLAY copy to grayscale if mode
        # is active.  Overlays (green/blue circles) are drawn
        # on this grayscale background — IR camera look.
        # ══════════════════════════════════════════════════════════
        try:
            mode = self._grayscale_mode_var.get()
            if mode == "off":
                image = self._current_image
            else:
                image = self._convert_display_frame(self._current_image.copy())

            canvas_w = self._canvas.winfo_width()
            canvas_h = self._canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                return
            h, w = image.shape[:2]
            scale = min(canvas_w / w, canvas_h / h, 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2
            self._display_scale = scale
            self._display_origin = (offset_x, offset_y)
            self._display_size = (new_w, new_h)

            display = cv2.resize(image, (new_w, new_h))

            result = self._current_result
            if result is not None and self._show_overlay.get():
                self._draw_overlay_scaled(display, result, scale)
            self._draw_manual_roi_overlay(display, scale)
            self._draw_manual_ring_overlay(display, scale)
            if self._show_debug_overlay.get():
                self._draw_debug_overlay(display, scale)
            self._show_image_fast(display, canvas_w, canvas_h, new_w, new_h)
        except Exception as exc:
            self.logger.exception("Display refresh error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")

    def _draw_welcome_screen(self) -> None:
        c = self._colors
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self._canvas.delete("all")
        self._canvas_image_id = None
        cx, cy = cw // 2, ch // 2
        self._canvas.create_text(
            cx,
            cy - 60,
            text="Medevplus IXcentai",
            fill=c.FG_PRIMARY,
            font=("Segoe UI", 22, "bold"),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy - 20,
            text="surgical grade",
            fill=c.ACCENT,
            font=("Segoe UI", 13),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy + 20,
            text="Open an image, video, or start the camera to begin",
            fill=c.FG_SECONDARY,
            font=("Segoe UI", 11),
            anchor="center",
        )
        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 11 of 12 — Updated welcome shortcuts
        # RECORDING — Updated shortcuts to include recording
        # ══════════════════════════════════════════════════════════
        shortcuts = (
            "Ctrl+O  Image    Ctrl+V  Video    Space  Pause"
            "    G  Grayscale    Ctrl+R  Record    Ctrl+Q  Quit"
        )
        # ══════════════════════════════════════════════════════════
        self._canvas.create_text(
            cx,
            cy + 60,
            text=shortcuts,
            fill=c.FG_TERTIARY,
            font=("Consolas", 9),
            anchor="center",
        )

    def _show_image(self, image_bgr: np.ndarray) -> None:
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
        h, w = image_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self._canvas.create_image(x, y, anchor=tk.NW, image=self._display_image)

    def _show_image_fast(
        self,
        image_bgr: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        img_w: int,
        img_h: int,
    ) -> None:
        """Display an already-resized BGR image, reusing the canvas item."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image = ImageTk.PhotoImage(pil_img)
        x = (canvas_w - img_w) // 2
        y = (canvas_h - img_h) // 2
        if self._canvas_image_id is not None:
            try:
                self._canvas.coords(self._canvas_image_id, x, y)
                self._canvas.itemconfig(
                    self._canvas_image_id, image=self._display_image
                )
                return
            except tk.TclError:
                self._canvas_image_id = None
        self._canvas.delete("all")
        self._canvas_image_id = self._canvas.create_image(
            x, y, anchor=tk.NW, image=self._display_image
        )

    @staticmethod
    def _scale_ellipse(e: Any, scale: float) -> SimpleNamespace:
        """Return ellipse namespace with coordinates scaled for display."""
        return SimpleNamespace(
            center_x=e.center_x * scale,
            center_y=e.center_y * scale,
            radius=e.radius * scale,
            semi_major=e.semi_major * scale,
            semi_minor=e.semi_minor * scale,
            angle_deg=e.angle_deg,
        )

    def _get_ellipse_intersection(
        self, ellipse: Any, px: float, py: float, dx: float, dy: float
    ) -> Tuple[float, float]:
        """Compute the intersection of a ray from (px, py) in direction (dx, dy) with an ellipse."""
        cx = ellipse.center_x
        cy = ellipse.center_y
        a = max(1.0, ellipse.semi_major)
        b = max(1.0, ellipse.semi_minor)
        angle_rad = math.radians(ellipse.angle_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Local coordinates of start point relative to ellipse center
        x_loc = (px - cx) * cos_a + (py - cy) * sin_a
        y_loc = -(px - cx) * sin_a + (py - cy) * cos_a

        # Local direction vector
        dx_loc = dx * cos_a + dy * sin_a
        dy_loc = -dx * sin_a + dy * cos_a

        # Quadratic equation coefficients for distance t: A * t^2 + 2 * B * t + C = 0
        A_coef = (dx_loc / a) ** 2 + (dy_loc / b) ** 2
        B_coef = (x_loc * dx_loc) / (a ** 2) + (y_loc * dy_loc) / (b ** 2)
        C_coef = (x_loc / a) ** 2 + (y_loc / b) ** 2 - 1.0

        disc = B_coef ** 2 - A_coef * C_coef
        if disc < 0 or A_coef == 0:
            # Fallback to a simple default bounding box boundary if math fails
            return px + dx * a, py + dy * b

        t = (-B_coef + math.sqrt(disc)) / A_coef
        return px + t * dx, py + t * dy

    def _draw_cross_section(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw intersecting horizontal/vertical cross section lines between pupil and limbus."""
        if not (
            result.pupil.detected
            and result.pupil.ellipse is not None
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            return

        p_ellipse = self._scale_ellipse(result.pupil.ellipse, scale)
        l_ellipse = self._scale_ellipse(result.limbus.ellipse, scale)

        p_cx = p_ellipse.center_x
        p_cy = p_ellipse.center_y

        # Calculate intersection points on the pupil boundary
        p_up_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, -1.0)
        p_down_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, 1.0)
        p_left_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, -1.0, 0.0)
        p_right_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 1.0, 0.0)

        l_cx = l_ellipse.center_x
        l_cy = l_ellipse.center_y

        # Calculate intersection points on the limbus boundary
        l_up_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, -1.0)
        l_down_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, 1.0)
        l_left_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, -1.0, 0.0)
        l_right_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 1.0, 0.0)

        # Colors (BGR)
        green_color = (0, 255, 0)      # Pupil green
        blue_color = (255, 100, 0)     # Limbus blue

        # Draw pupil cross
        cv2.line(out, (int(round(p_left_pt[0])), int(round(p_left_pt[1]))), (int(round(p_right_pt[0])), int(round(p_right_pt[1]))), green_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(p_up_pt[0])), int(round(p_up_pt[1]))), (int(round(p_down_pt[0])), int(round(p_down_pt[1]))), green_color, 1, cv2.LINE_AA)

        # Draw limbus cross
        cv2.line(out, (int(round(l_left_pt[0])), int(round(l_left_pt[1]))), (int(round(l_right_pt[0])), int(round(l_right_pt[1]))), blue_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(l_up_pt[0])), int(round(l_up_pt[1]))), (int(round(l_down_pt[0])), int(round(l_down_pt[1]))), blue_color, 1, cv2.LINE_AA)

        # Draw ASCII degree labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_sz = max(0.3, 0.4 * scale)
        lbl_color = (220, 220, 220)

        # UP: 270 degrees
        up_x, up_y = int(round(l_up_pt[0])), int(round(l_up_pt[1]))
        cv2.putText(out, "270", (up_x - int(10 * scale), up_y - int(5 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # DOWN: 90 degrees
        down_x, down_y = int(round(l_down_pt[0])), int(round(l_down_pt[1]))
        cv2.putText(out, "90", (down_x - int(7 * scale), down_y + int(12 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # LEFT: 0 degrees
        left_x, left_y = int(round(l_left_pt[0])), int(round(l_left_pt[1]))
        cv2.putText(out, "0", (left_x - int(15 * scale), left_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # RIGHT: 180 degrees
        right_x, right_y = int(round(l_right_pt[0])), int(round(l_right_pt[1]))
        cv2.putText(out, "180", (right_x + int(5 * scale), right_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

    def _draw_overlay_scaled(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw overlays on an already-resized image with scaled coords."""
        h, w = out.shape[:2]
        cal = result.calibration

        ring_status = getattr(result, "ring_status", "unknown")
        if ring_status == "ring_present":
            ring_center = getattr(result, "ring_center", None)
            ring_radius = getattr(result, "ring_radius", None)
            ring_contour = getattr(result, "ring_contour", None)
            if ring_center is not None and ring_radius is not None:
                cx = int(round(ring_center[0] * scale))
                cy = int(round(ring_center[1] * scale))
                rr = int(round(ring_radius * scale))
                if ring_contour is not None and len(ring_contour) >= 5:
                    scaled = np.round(ring_contour.astype(np.float32) * scale).astype(np.int32)
                    cv2.drawContours(out, [scaled], -1, (0, 0, 255), 2)
                else:
                    cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
                if self._show_ring_center.get():
                    _base = max(4, int(10 * scale))
                    _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                    if _cal_mm_per_px > 0:
                        _ring_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                    else:
                        _ring_cross_size = int(max(12, min(_base, 26)))
                    cv2.drawMarker(
                        out,
                        (cx, cy),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        _ring_cross_size,
                        2,
                        cv2.LINE_AA,
                    )
                if self._show_measurements.get():
                    label = f"R={ring_radius * 2.0:.0f}px"
                    if cal.calibrated:
                        label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                    cv2.putText(
                        out,
                        label,
                        (cx + 10, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, 0.45 * scale),
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw limbus first (larger) so pupil renders on top
        if (
            self._show_limbus.get()
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            e_orig = result.limbus.ellipse
            e = self._scale_ellipse(e_orig, scale)
            limbus_color = (255, 100, 0)
            limbus_alpha = self._limbus_fill_alpha_var.get() / 100.0
            if limbus_alpha > 0:
                self._draw_filled_structure(out, e, limbus_color, limbus_alpha)
            ct = self._draw_structure(out, e, limbus_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), limbus_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    dia_mm = dia_px * cal.mm_per_px
                    smaj_mm = e_orig.semi_major * cal.mm_per_px
                    smin_mm = e_orig.semi_minor * cal.mm_per_px
                    label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    limbus_color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw pupil second so it always appears on top of the limbus fill
        if (
            self._show_pupil.get()
            and result.pupil.detected
            and result.pupil.ellipse is not None
        ):
            e_orig = result.pupil.ellipse
            e = self._scale_ellipse(e_orig, scale)
            pupil_color = (0, 255, 0)
            pupil_alpha = self._pupil_fill_alpha_var.get() / 100.0
            if pupil_alpha > 0:
                self._draw_filled_structure(out, e, pupil_color, pupil_alpha)
            ct = self._draw_structure(out, e, pupil_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), pupil_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    pupil_color,
                    1,
                    cv2.LINE_AA,
                )

        cc = getattr(result, "corneal_center", None)
        if (
            self._show_centers.get()
            and cc is not None
            and getattr(cc, "valid", False)
            and getattr(cc, "center_px", None)
        ):
            center_pt = (
                int(round(cc.center_px[0] * scale)),
                int(round(cc.center_px[1] * scale)),
            )
            # Marker size is specified in *display* pixels. To keep the marker
            # size physically meaningful, we scale it using the current
            # calibration when available.
            #
            # Marker size is specified in *display* pixels.
            # Keep the marker size physically meaningful by converting
            # a desired physical offset (+3mm relative to the previous
            # baseline marker) into image pixels using calibration.
            _base = max(4, int(10 * scale))
            cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
            if cal_mm_per_px > 0:
                cursor_size = int(max(10, round(_base + (0.5 / cal_mm_per_px) * scale)))
            else:
                cursor_size = int(max(12, min(_base, 26)))
            cv2.drawMarker(
                out,
                center_pt,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                cursor_size,
                2,
                cv2.LINE_AA,
            )
            if self._show_measurements.get():
                ref_name = getattr(result, "corneal_reference_source", "cornea")
                cv2.putText(
                    out,
                    f"Corneal Centre [{ref_name}]",
                    (center_pt[0] + 12, center_pt[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.46 * scale),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self._show_offset.get() and result.has_both:
            p = result.pupil.ellipse
            p_pt = (int(round(p.center_x * scale)), int(round(p.center_y * scale)))
            if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
                ref_pt = (
                    int(round(cc.center_px[0] * scale)),
                    int(round(cc.center_px[1] * scale)),
                )
                dx = p.center_x - cc.center_px[0]
                dy = p.center_y - cc.center_px[1]
            else:
                l = result.limbus.ellipse
                ref_pt = (int(round(l.center_x * scale)), int(round(l.center_y * scale)))
                dx = p.center_x - l.center_x
                dy = p.center_y - l.center_y
            cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
            if self._show_centers.get():
                _base = max(4, int(10 * scale))
                _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                if _cal_mm_per_px > 0:
                    _offset_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                else:
                    _offset_cross_size = int(max(12, min(_base, 26)))
                cv2.drawMarker(
                    out,
                    ref_pt,
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    _offset_cross_size,
                    2,
                    cv2.LINE_AA,
                )
            if self._show_measurements.get():
                offset_px = math.hypot(dx, dy)
                mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
                label = f"{offset_px:.1f}px"
                if cal.calibrated:
                    label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
                font_scale = max(0.25, 0.4 * scale)
                cv2.putText(
                    out,
                    label,
                    (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self._draw_cross_section(out, result, scale)

        quality = (
            result.overall_quality.value
            if hasattr(result.overall_quality, "value")
            else str(result.overall_quality)
        )
        color_map = {
            "SURGICAL": (0, 230, 118),
            "CLINICAL": (246, 182, 41),
            "RESEARCH": (38, 167, 255),
            "INSUFFICIENT": (80, 83, 239),
            "NO_DETECTION": (97, 97, 97),
        }
        badge_color = color_map.get(quality, (128, 128, 128))
        font_scale_q = max(0.4, 0.7 * scale)
        cv2.putText(
            out,
            f"{quality} ({result.overall_confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_q,
            badge_color,
            2,
        )
        font_scale_t = max(0.3, 0.5 * scale)
        cv2.putText(
            out,
            f"{result.metadata.processing_time_ms:.0f}ms",
            (w - max(80, int(100 * scale)), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_t,
            (180, 180, 180),
            1,
        )
        if self._last_opt_stats.get("overload_active"):
            label = "OVERLOAD PROTECTION"
            org = (10, 58)
            cv2.putText(
                out,
                label,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.55 * scale),
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                label,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.55 * scale),
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 12 of 12 — Grayscale mode badge on image
        # ══════════════════════════════════════════════════════════
        # (Removed) Grayscale mode badge text overlay (top-right).
        # Grayscale processing (OFF/AUTO/FORCE) and overlays remain active; 
        # only the on-image text label was removed to reduce clutter.
        mode = self._grayscale_mode_var.get()
        _ = mode  # keep reference to avoid unused variable warnings
        # ══════════════════════════════════════════════════════════

        font_scale_a = max(0.25, 0.4 * scale)
        for i, alert in enumerate(result.alerts[:3]):
            cv2.putText(
                out,
                alert[:80],
                (10, h - 15 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_a,
                (0, 100, 255),
                1,
            )

        self._draw_ruler_overlay(out, scale)

    def _draw_debug_overlay(self, out: np.ndarray, scale: float) -> None:

        stats = self._last_opt_stats
        if not stats:
            return
        h, w = out.shape[:2]
        pad = max(8, int(10 * scale))
        line_gap = max(16, int(18 * scale))
        font_scale = max(0.32, 0.42 * scale)
        lines = [
            f"Preset: {self._performance_preset_var.get().replace('_', ' ').title()}",
            f"Pipeline: {stats.get('backend', self._pipeline_var.get())}",
            f"Latency avg: {float(stats.get('latency_avg_ms', 0.0) or 0.0):.1f} ms",
            f"Proc avg: {float(stats.get('processing_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI avg: {float(stats.get('roi_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI mode: {str(stats.get('roi_mode', 'off')).title()}",
            f"Tracking: {self._tracking_state_var.get() or '---'}",
            (
                "Adaptive quality: "
                + (
                    f"ON (stable={int(stats.get('stable_tracking_streak', 0))}, "
                    f"skips={int(stats.get('quality_check_skips', 0))})"
                    if stats.get("adaptive_quality_active")
                    else "OFF"
                )
            ),
            (
                f"Dropped/Stale: {int(stats.get('dropped_frames', 0))}/"
                f"{int(stats.get('stale_frames', 0))}"
            ),
            (
                "Overload protection: "
                + (
                    f"ACTIVE (reuse={int(stats.get('cached_reuse_total', 0))})"
                    if stats.get("overload_active")
                    else f"Idle (reuse={int(stats.get('cached_reuse_total', 0))})"
                )
            ),
        ]
        box_width = max(220, int(w * 0.34))
        box_height = pad * 2 + line_gap * len(lines)
        x0 = max(0, w - box_width - pad)
        y0 = max(0, h - box_height - pad)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (18, 18, 18), -1)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (70, 70, 70), 1)
        for idx, line in enumerate(lines):
            y = y0 + pad + line_gap * (idx + 1) - 4
            cv2.putText(
                out,
                line,
                (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_roi_overlay(self, out: np.ndarray, scale: float) -> None:
        roi = (
            self._roi_preview
            if self._roi_preview is not None
            else self._active_manual_roi()
        )
        if roi is None:
            return
        cx = int(round(roi["center_x"] * scale))
        cy = int(round(roi["center_y"] * scale))
        radius = max(1, int(round(roi["radius"] * scale)))
        is_editing = self._roi_preview is not None and self._roi_edit_active
        color = (0, 255, 255) if is_editing else (0, 220, 255)
        original = out.copy()
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
        shaded = out.copy()
        shaded[:] = (15, 15, 15)
        outside = cv2.bitwise_not(mask)
        out[:] = cv2.addWeighted(out, 0.45, shaded, 0.55, 0.0)
        inside_original = cv2.bitwise_and(original, original, mask=mask)
        outside_dimmed = cv2.bitwise_and(out, out, mask=outside)
        out[:] = cv2.add(inside_original, outside_dimmed)
        cv2.circle(out, (cx, cy), radius, color, 2, cv2.LINE_AA)
        if is_editing:
            cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)
        
        dia_px = roi["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"
        
        label = f"ROI {dia_str} (Enter=lock)" if is_editing else f"ROI {dia_str}"
        font_scale = max(0.4, 0.5 * scale)
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = max(10, cx - radius - text_w - 10)
        if is_editing:
            cv2.putText(
                out,
                label,
                (text_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        
            caption = "ROI Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_ring_overlay(self, out: np.ndarray, scale: float) -> None:
        ring = (
            self._ring_preview
            if self._ring_preview is not None
            else self._active_manual_ring()
        )
        if ring is None:
            return
        cx = int(round(ring["center_x"] * scale))
        cy = int(round(ring["center_y"] * scale))
        radius = max(1, int(round(ring["radius"] * scale)))
        is_editing = self._ring_preview is not None and self._ring_edit_active
        if (
            not is_editing
            and self._current_result is not None
            and getattr(self._current_result, "ring_status", "unknown") == "ring_present"
        ):
            return
        color = (40, 110, 255) if is_editing else (0, 0, 255)
        thickness = 2 if is_editing else 3
        cv2.circle(out, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)
        
        dia_px = ring["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"
            
        label = f"Manual Ring {dia_str} (Enter=lock)" if is_editing else f"Manual Ring {dia_str}"
        cv2.putText(
            out,
            label,
            (cx + 10, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, 0.45 * scale),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(out, (handle_x, handle_y), max(5, int(7 * scale)), (20, 20, 20), 1)
        
        if is_editing:
            caption = "Ring Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_ruler_overlay(self, out: np.ndarray, scale: float) -> None:
        if not getattr(self, "_ruler_calibration_active", False) and not getattr(self, "_ruler_points", None):
            return
        pts = getattr(self, "_ruler_points", [])
        color = (0, 255, 255)
        for i, pt in enumerate(pts):
            cx = int(round(pt[0] * scale))
            cy = int(round(pt[1] * scale))
            cv2.circle(out, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 9, color, 2, cv2.LINE_AA)
            cv2.putText(
                out,
                f"P{i+1}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.5 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

        if len(pts) >= 2:
            p1 = (int(round(pts[0][0] * scale)), int(round(pts[0][1] * scale)))
            p2 = (int(round(pts[1][0] * scale)), int(round(pts[1][1] * scale)))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            dist_px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                out,
                f"{dist_px:.1f} px = {known_mm:.1f} mm ({dist_px/known_mm:.2f} px/mm)",
                (mid_x + 10, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.52 * scale),
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod

    def _draw_filled_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Fill the circle/ellipse area with a semi-transparent colour overlay."""
        if alpha <= 0.0:
            return
        overlay = out.copy()
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        ratio = (ellipse_data.semi_minor / ellipse_data.semi_major
                 if ellipse_data.semi_major > 0 else 1.0)
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(overlay, ct, r, color, -1, cv2.LINE_AA)
        else:
            axes = (int(round(ellipse_data.semi_major)),
                    int(round(ellipse_data.semi_minor)))
            cv2.ellipse(overlay, ct, axes, int(round(ellipse_data.angle_deg)),
                        0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    @staticmethod
    def _draw_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> Tuple[int, int]:
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        if ellipse_data.semi_major > 0:
            ratio = ellipse_data.semi_minor / ellipse_data.semi_major
        else:
            ratio = 1.0
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(out, ct, r, color, thickness, cv2.LINE_AA)
        else:
            axes = (
                int(round(ellipse_data.semi_major)),
                int(round(ellipse_data.semi_minor)),
            )
            angle = int(round(ellipse_data.angle_deg))
            cv2.ellipse(out, ct, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        return ct

    def _draw_overlay(self, image: np.ndarray, result: Any) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        cal = result.calibration

        if (
            self._show_pupil.get()
            and result.pupil.detected
            and result.pupil.ellipse is not None
        ):
            e = result.pupil.ellipse
            pupil_color = (0, 255, 0)
            ct = self._draw_structure(out, e, pupil_color)
            if self._show_centers.get():
                cv2.circle(out, ct, 4, pupil_color, -1)
            if self._show_measurements.get():
                dia_px = e.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
                ft = getattr(e, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    pupil_color,
                    1,
                    cv2.LINE_AA,
                )

        if (
            self._show_limbus.get()
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            e = result.limbus.ellipse
            limbus_color = (255, 100, 0)
            ct = self._draw_structure(out, e, limbus_color)
            if self._show_centers.get():
                cv2.circle(out, ct, 4, limbus_color, -1)
            if self._show_measurements.get():
                dia_px = e.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    dia_mm = dia_px * cal.mm_per_px
                    smaj_mm = e.semi_major * cal.mm_per_px
                    smin_mm = e.semi_minor * cal.mm_per_px
                    label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
                ft = getattr(e, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    limbus_color,
                    1,
                    cv2.LINE_AA,
                )

        ring_status = getattr(result, "ring_status", "unknown")
        if ring_status == "ring_present":
            ring_center = getattr(result, "ring_center", None)
            ring_radius = getattr(result, "ring_radius", None)
            ring_contour = getattr(result, "ring_contour", None)
            if ring_center is not None and ring_radius is not None:
                cx = (int(round(ring_center[0])))
                cy = (int(round(ring_center[1])))
                rr = int(round(ring_radius))
                if ring_contour is not None and len(ring_contour) >= 5:
                    cv2.drawContours(out, [ring_contour.astype(np.int32)], -1, (0, 0, 255), 2)
                else:
                    cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
                if self._show_ring_center.get():
                    _base = max(4, int(10))
                    _ring_cross_size = int(max(12, min(_base, 26)))
                    cv2.drawMarker(
                        out,
                        (cx, cy),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        _ring_cross_size,
                        2,
                        cv2.LINE_AA,
                    )
                if self._show_measurements.get():
                    label = f"R={ring_radius * 2.0:.0f}px"
                    if cal.calibrated:
                        label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                    cv2.putText(
                        out,
                        label,
                        (cx + 10, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if self._show_offset.get() and result.has_both:
            p = result.pupil.ellipse
            p_pt = (int(round(p.center_x)), int(round(p.center_y)))
            cc = getattr(result, "corneal_center", None)
            if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
                ref_pt = (
                    int(round(cc.center_px[0])),
                    int(round(cc.center_px[1])),
                )
                dx = p.center_x - cc.center_px[0]
                dy = p.center_y - cc.center_px[1]
            else:
                l = result.limbus.ellipse
                ref_pt = (int(round(l.center_x)), int(round(l.center_y)))
                dx = p.center_x - l.center_x
                dy = p.center_y - l.center_y
            cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
            if self._show_centers.get():
                cv2.drawMarker(out, ref_pt, (10, 10, 10), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                cv2.drawMarker(out, ref_pt, (255, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            if self._show_measurements.get():
                offset_px = math.hypot(dx, dy)
                mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
                label = f"{offset_px:.1f}px"
                if cal.calibrated:
                    label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
                cv2.putText(
                    out,
                    label,
                    (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self._draw_cross_section(out, result, 1.0)

        quality = (
            result.overall_quality.value
            if hasattr(result.overall_quality, "value")
            else str(result.overall_quality)
        )
        color_map = {
            "SURGICAL": (0, 230, 118),
            "CLINICAL": (246, 182, 41),
            "RESEARCH": (38, 167, 255),
            "INSUFFICIENT": (80, 83, 239),
            "NO_DETECTION": (97, 97, 97),
        }
        badge_color = color_map.get(quality, (128, 128, 128))
        cv2.putText(
            out,
            f"{quality} ({result.overall_confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            badge_color,
            2,
        )
        cv2.putText(
            out,
            f"{result.metadata.processing_time_ms:.0f}ms",
            (w - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        # (Hidden) GRAYSCALE GUI 12 of 12 — Grayscale mode badge on image
        # Removed to avoid showing overlay text when user loads an image.


        for i, alert in enumerate(result.alerts[:3]):
            cv2.putText(
                out,
                alert[:80],
                (10, h - 15 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 100, 255),
                1,
            )

        return out

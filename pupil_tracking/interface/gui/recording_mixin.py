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


class RecordingMixin:
    _CAPTURE_GLYPH_MAP = {
        "°": " deg",   # ° degree
        "×": "x",      # × multiplication
        "—": "-",      # — em dash
        "–": "-",      # – en dash
        "→": "->",     # → arrow
        "µ": "u",      # µ micro
        "≥": ">=",     # ≥
        "≤": "<=",     # ≤
        "±": "+/-",    # ±
        "•": "*",      # • bullet
        "⚠": "!",      # ⚠ warning
        "⚡": "",       # ⚡ lightning
        "✓": "OK",     # ✓ check
        "✗": "X",      # ✗ cross
    }

    def _choose_recording_path(self) -> Optional[str]:
        """Ask user where to save the recording."""
        default_name = self._recording_default_path_var.get()
        if not default_name:
            default_name = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path = filedialog.asksaveasfilename(
            title="Save Recording As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video (H.264)", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("All files", "*.*"),
            ],
            initialfile=default_name,
        )
        return path if path else None

    def _start_recording(self) -> None:
        """Start recording the current view (video/camera feed)."""
        if self._recorder.is_recording:
            return

        if self._current_image is None:
            messagebox.showinfo(
                "No Source",
                "Start a video or camera before recording.",
            )
            return

        path = self._choose_recording_path()
        if not path:
            return

        # Prepare a frame that includes the measurements panel so the
        # recording captures the full on-screen layout (image + table).
        display_image = self._prepare_recording_frame(
            self._current_image,
            self._current_result,
        )
        composite = self._compose_capture_frame(display_image, self._current_result)
        h, w = composite.shape[:2]

        target_fps = 30.0
        if self._video_cap is not None:
            fps = self._video_cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps <= 120:
                target_fps = fps

        if not self._recorder.start(path, w, h, target_fps):
            messagebox.showerror(
                "Recording Error",
                f"Cannot start recording. Check codec support.\nPath: {path}",
            )
            return

        self._recording_path = path
        self._recording_default_path_var.set(
            f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        self._update_recording_ui(started=True)
        self._status_var.set(f"Recording started → {path}")
        self._start_recording_timer()

    def _stop_recording(self) -> None:
        """Stop recording and save the video file."""
        if not self._recorder.is_recording:
            return

        path = self._recorder.stop()
        self._stop_recording_timer()

        elapsed = (
            self._recorder.elapsed_time
            if hasattr(self._recorder, "elapsed_time")
            else 0
        )
        frame_count = (
            self._recorder.frame_count if hasattr(self._recorder, "frame_count") else 0
        )

        self._update_recording_ui(started=False)
        self._status_var.set(
            f"Recording saved: {frame_count} frames in {elapsed:.1f}s → {path or 'unknown'}"
        )

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_recorder_status(self, status: dict) -> None:
        """Callback from FrameRecorder for status updates."""
        if status.get("is_recording"):
            elapsed = status.get("elapsed_time", 0)
            fps = status.get("fps", 0)
            dropped = status.get("dropped_frames", 0)
            frames = status.get("frame_count", 0)

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  REC {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_fps_var.set(fps)
            self._recording_dropped_var.set(dropped)
        else:
            self._recording_indicator.config(text="  --:--  ", foreground="#616161")

    def _start_recording_timer(self) -> None:
        """Start the recording indicator update timer."""
        self._stop_recording_timer()
        self._update_recording_timer()

    def _update_recording_timer(self) -> None:
        """Periodically update the recording indicator."""
        if self._recorder.is_recording:
            elapsed = self._recorder.elapsed_time
            fps = self._recorder.frame_count / elapsed if elapsed > 0 else 0
            dropped = self._recorder.dropped_frames

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  REC {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_timer_id = self.root.after(
                500, self._update_recording_timer
            )
        else:
            self._stop_recording_timer()

    def _stop_recording_timer(self) -> None:
        """Stop the recording indicator update timer."""
        if self._recording_timer_id is not None:
            self.root.after_cancel(self._recording_timer_id)
            self._recording_timer_id = None

    def _update_recording_ui(self, started: bool = False) -> None:
        """Update recording button state."""
        if hasattr(self, "_rec_btn"):
            if started or self._recorder.is_recording:
                self._rec_btn.config(text="⏹ Stop Rec")
            else:
                self._rec_btn.config(text="⏺ Start Rec")

    def _write_frame_to_recorder(self, frame: np.ndarray) -> None:
        """Write a frame to the recorder (non-blocking)."""
        # If recording, ensure we write the full composed UI (image +
        # measurements panel). The provided `frame` may be just the
        # image area, so construct the composite using current state.
        try:
            if self._recorder.is_recording and self._current_image is not None:
                display_image = self._prepare_recording_frame(
                    self._current_image, self._current_result
                )
                composite = self._compose_capture_frame(display_image, self._current_result)
                self._recorder.write(composite)
                return
        except Exception:
            # Fall back to naive write if composition fails
            pass

        self._recorder.write(frame)

    def _prepare_recording_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """Prepare a frame that mirrors the current on-screen display for recording."""
        mode = self._grayscale_mode_var.get()
        if mode == "off":
            image = frame.copy()
        else:
            image = self._convert_display_frame(frame.copy())

        if result is not None and self._show_overlay.get():
            self._draw_overlay_scaled(image, result, 1.0)

        self._draw_manual_roi_overlay(image, 1.0)
        self._draw_manual_ring_overlay(image, 1.0)
        if self._show_debug_overlay.get():
            self._draw_debug_overlay(image, 1.0)

        return image

    @staticmethod
    def _hex_to_bgr(value: str) -> Tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            return (200, 200, 200)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (b, g, r)

    def _measurement_capture_sections(self) -> List[Tuple[str, Tuple[int, int, int], List[Tuple[str, str]]]]:
        return [
            (
                "PUPIL",
                self._hex_to_bgr(self._colors.PUPIL),
                [
                    ("Center", self._pv["center"].get()),
                    ("Diameter (px)", self._pv["diameter_px"].get()),
                    ("Diameter (mm)", self._pv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._pv["semi_major"].get()),
                    ("Semi-Major (mm)", self._pv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._pv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._pv["semi_minor_mm"].get()),
                    ("Angle", self._pv["angle"].get()),
                    ("Fit Type", self._pv["fit_type"].get()),
                    ("Confidence", self._pv["confidence"].get()),
                    ("Quality", self._pv["quality"].get()),
                ],
            ),
            (
                "LIMBUS",
                self._hex_to_bgr(self._colors.LIMBUS),
                [
                    ("Center", self._lv["center"].get()),
                    ("Diameter (px)", self._lv["diameter_px"].get()),
                    ("Diameter (mm)", self._lv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._lv["semi_major"].get()),
                    ("Semi-Major (mm)", self._lv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._lv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._lv["semi_minor_mm"].get()),
                    ("Angle", self._lv["angle"].get()),
                    ("Fit Type", self._lv["fit_type"].get()),
                    ("Confidence", self._lv["confidence"].get()),
                    ("Quality", self._lv["quality"].get()),
                ],
            ),
            (
                "CORNEAL OFFSET",
                self._hex_to_bgr(self._colors.OFFSET),
                [
                    ("Corneal Centre", self._ov["corneal_center"].get()),
                    ("Offset (px)", self._ov["offset_px"].get()),
                    ("Offset (mm)", self._ov["offset_mm"].get()),
                    ("Offset dX,dY px", self._ov["offset_vec_px"].get()),
                    ("Offset dX,dY mm", self._ov["offset_vec_mm"].get()),
                    ("Offset Angle", self._ov["offset_angle"].get()),
                    ("Pupil/Limbus", self._ov["pupil_limbus_ratio"].get()),
                ],
            ),
            (
                "CALIBRATION",
                self._hex_to_bgr(self._colors.CALIBRATION),
                [
                    ("Source", self._cv_vars["source"].get()),
                    ("px/mm", self._cv_vars["scale_px"].get()),
                    ("mm/px", self._cv_vars["scale_mm"].get()),
                    ("Reference", self._cv_vars["reference"].get()),
                ],
            ),
            (
                "PROCESSING",
                self._hex_to_bgr(self._colors.PROCESSING),
                [
                    ("Proc. Time", self._proc_time_var.get()),
                    ("Latency", self._latency_var.get()),
                    ("Latency Avg", self._latency_avg_var.get()),
                    ("Dropped/Stale", self._drop_var.get()),
                    ("Tracking", self._tracking_state_var.get()),
                    ("FPS", self._fps_var.get()),
                    ("Frame", self._frame_var.get()),
                    ("Image Size", self._image_size_var.get()),
                    ("Pipeline", self._pipeline_var.get()),
                    ("Grayscale", self._gray_mode_var_display.get()),
                ],
            ),
        ]

    def _ascii_for_capture(self, text: str) -> str:
        """Sanitize a UI string so cv2.putText renders it without '?' glyphs.

        Replaces known symbols with ASCII equivalents, then drops any
        remaining non-ASCII codepoint so nothing renders as a question mark.
        """
        if not text:
            return text
        for uni, ascii_rep in self._CAPTURE_GLYPH_MAP.items():
            if uni in text:
                text = text.replace(uni, ascii_rep)
        # Drop any leftover non-ASCII so OpenCV never emits '?'
        if any(ord(ch) > 127 for ch in text):
            text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def _render_measurements_capture(self, height: int, width: int) -> np.ndarray:
        panel = np.full((height, width, 3), self._hex_to_bgr(self._colors.BG_SECONDARY), dtype=np.uint8)
        cv2.rectangle(
            panel,
            (0, 0),
            (width - 1, height - 1),
            self._hex_to_bgr(self._colors.BORDER),
            1,
        )
        pad = max(10, height // 72)
        gutter = max(10, width // 42)
        inner_w = width - pad * 2
        col_w = max(220, (inner_w - gutter) // 2)
        title_font = max(0.42, min(0.72, height / 900.0))
        body_font = max(0.34, min(0.54, height / 1080.0))
        line_h = max(16, int(height / 36))
        row_gap = max(3, int(line_h * 0.2))
        section_gap = max(8, int(line_h * 0.55))
        summary_box_h = max(62, int(height * 0.09))
        summary_gap = max(8, gutter // 2)
        summary_w = max(120, (inner_w - summary_gap) // 2)
        fg_primary = self._hex_to_bgr(self._colors.FG_PRIMARY)
        fg_secondary = self._hex_to_bgr(self._colors.FG_SECONDARY)
        card_bg = self._hex_to_bgr(self._colors.BG_TERTIARY)
        quality_color = self._hex_to_bgr(
            _QUALITY_COLORS.get(self._summary_quality_var.get(), self._colors.FG_PRIMARY)
        )
        tracking_color = self._hex_to_bgr(
            {
                "Tracking Stable": self._colors.SURGICAL,
                "Tracking Acquiring": self._colors.CLINICAL,
                "Tracking Degraded": self._colors.RESEARCH,
                "No Detection": self._colors.INSUFFICIENT,
                "Ready": self._colors.ACCENT,
                "Waiting": self._colors.FG_SECONDARY,
            }.get(self._summary_tracking_var.get(), self._colors.FG_PRIMARY)
        )

        summaries = [
            ("QUALITY", self._summary_quality_var.get(), quality_color),
            ("TRACKING", self._summary_tracking_var.get(), tracking_color),
            ("LATENCY", self._summary_latency_var.get(), fg_primary),
            ("PIPELINE", self._summary_pipeline_var.get(), fg_primary),
        ]
        for idx, (label, value, color) in enumerate(summaries):
            row = idx // 2
            col = idx % 2
            x0 = pad + col * (summary_w + summary_gap)
            y0 = pad + row * (summary_box_h + summary_gap)
            x1 = min(width - pad, x0 + summary_w)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), card_bg, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), self._hex_to_bgr(self._colors.BORDER), 1)
            cv2.putText(panel, self._ascii_for_capture(label), (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg_secondary, 1, cv2.LINE_AA)
            cv2.putText(panel, self._ascii_for_capture(value or "---"), (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        sections = self._measurement_capture_sections()
        left_sections = sections[:2]
        right_sections = sections[2:]
        start_y = pad * 2 + summary_box_h * 2 + summary_gap

        def draw_section_column(items, x_start):
            y = start_y
            for title, accent, rows in items:
                box_h = max(60, 34 + len(rows) * (line_h + row_gap))
                if y + box_h > height - pad:
                    box_h = max(40, height - pad - y)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), card_bg, -1)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), self._hex_to_bgr(self._colors.BORDER), 1)
                cv2.putText(panel, self._ascii_for_capture(title), (x_start + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, title_font, accent, 2, cv2.LINE_AA)
                row_y = y + 48
                for label, value in rows:
                    if row_y > y + box_h - 8:
                        break
                    clean_label = self._ascii_for_capture(label.replace("_", " ").title())
                    clean_value = self._ascii_for_capture((value or "---").replace("\n", " "))
                    if len(clean_value) > 32:
                        clean_value = clean_value[:29] + "..."
                    cv2.putText(panel, clean_label, (x_start + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_secondary, 1, cv2.LINE_AA)
                    text_size = cv2.getTextSize(clean_value, cv2.FONT_HERSHEY_SIMPLEX, body_font, 1)[0]
                    value_x = max(x_start + 140, x_start + col_w - 10 - text_size[0])
                    cv2.putText(panel, clean_value, (value_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_primary, 1, cv2.LINE_AA)
                    row_y += line_h + row_gap
                y += box_h + section_gap

        draw_section_column(left_sections, pad)
        draw_section_column(right_sections, pad + col_w + gutter)
        return panel

    def _compose_capture_frame(self, image: np.ndarray, result: Any) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        panel_w = max(700, int(img_w * 0.62))
        panel = self._render_measurements_capture(img_h, panel_w)
        divider = np.full((img_h, 3, 3), self._hex_to_bgr(self._colors.BORDER), dtype=np.uint8)
        return np.concatenate([image, divider, panel], axis=1)

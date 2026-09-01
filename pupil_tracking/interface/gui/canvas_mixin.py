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


class CanvasMixin:
    def _begin_roi_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag a circular ROI on the image")
            return
        if self._ring_edit_active:
            self._cancel_ring_selection()
        self._roi_edit_active = True
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        active_roi = self._active_manual_roi()
        self._roi_original_before_edit = (
            dict(active_roi) if active_roi is not None else None
        )
        if active_roi is not None:
            self._roi_preview = dict(active_roi)
        else:
            h, w = self._current_image.shape[:2]
            radius = 939.0 / 2.0
            self._roi_preview = {
                "center_x": w / 2.0,
                "center_y": h / 2.0,
                "radius": radius,
                "frame_width": float(w),
                "frame_height": float(h),
            }
        self._canvas.configure(cursor="tcross")
        if hasattr(self, "_roi_btn"):
            self._roi_btn.config(text="Edit ROI")
        self._roi_status_var.set("Manual ROI: Editing")
        self._status_var.set(
            "ROI edit mode: drag inside to move, drag rim to resize, Enter to apply, Esc to cancel"
        )
        self._refresh_display()

    def _begin_ring_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag the docked red ring on the image")
            return
        if self._roi_edit_active:
            self._cancel_roi_selection()
        self._ring_edit_active = True
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        active_ring = self._active_manual_ring()
        self._ring_original_before_edit = (
            dict(active_ring) if active_ring is not None else None
        )
        if active_ring is not None:
            self._ring_preview = dict(active_ring)
        else:
            self._ring_preview = self._suggest_manual_ring_preview()
        self._canvas.configure(cursor="tcross")
        if hasattr(self, "_ring_btn"):
            self._ring_btn.config(text="Edit Ring")
        self._ring_status_var.set("Manual Ring: Editing")
        self._status_var.set(
            "Ring edit mode: use only on docked frames, drag circle to match red dots, Enter to lock"
        )
        self._refresh_display()

    def _clear_manual_roi(self) -> None:
        self._manual_roi = None
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_roi_btn"):
            self._roi_btn.config(text="Set ROI")
        self._roi_status_var.set("Manual ROI: Off")
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_roi()
        self._refresh_display()

    def _clear_manual_ring(self) -> None:
        self._manual_ring = None
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_ring_btn"):
            self._ring_btn.config(text="Set Ring")
        self._ring_status_var.set("Manual Ring: Off")
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_ring()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._refresh_display()

    def _on_canvas_press(self, event: Any) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._handle_ruler_canvas_click(event)
            return
        if self._ring_edit_active:
            self._handle_ring_canvas_press(event)
            return
        if not self._roi_edit_active:
            return
        point = self._canvas_to_image_point(event.x, event.y)

        if point is None or self._current_image is None:
            return
        if self._roi_preview is None:
            h, w = self._current_image.shape[:2]
            self._roi_preview = {
                "center_x": point[0],
                "center_y": point[1],
                "radius": 939.0 / 2.0,
                "frame_width": float(w),
                "frame_height": float(h),
            }

        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        radius = self._roi_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.18)

        if abs(distance - radius) <= rim_threshold:
            self._roi_drag_mode = "resize"
        elif distance < radius:
            self._roi_drag_mode = "move"
            self._roi_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._roi_drag_mode = "resize"
            self._roi_preview["center_x"] = point[0]
            self._roi_preview["center_y"] = point[1]
            self._roi_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _on_canvas_drag(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_drag(event)
            return
        if (
            not self._roi_edit_active
            or self._roi_drag_mode is None
            or self._roi_preview is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        h, w = self._current_image.shape[:2]
        if self._roi_drag_mode == "move":
            radius = self._roi_preview["radius"]
            cx = point[0] - self._roi_drag_offset[0]
            cy = point[1] - self._roi_drag_offset[1]
            self._roi_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._roi_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._roi_preview["center_x"]
            cy = self._roi_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _on_canvas_release(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_release(event)
            return
        if not self._roi_edit_active:
            return
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._roi_preview is not None:
            self._status_var.set(
                "ROI ready. Drag to refine, press Enter to apply, or Esc to cancel"
            )
        self._refresh_display()

    def _confirm_active_selection(self, event: Any = None) -> None:
        if self._ring_edit_active:
            self._confirm_ring_selection(event)
            return
        self._confirm_roi_selection(event)

    def _confirm_roi_selection(self, event: Any = None) -> None:
        if not self._roi_edit_active:
            return
        preview = self._roi_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("ROI selection cancelled")
            self._cancel_roi_selection()
            return
        self._manual_roi = dict(preview)
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_roi_btn"):
            self._roi_btn.config(text="Set ROI")
        self._roi_status_var.set(
            f"Manual ROI: On ({int(round(self._manual_roi['radius']))} px)"
        )
        self._status_var.set("Manual ROI applied to live detection")
        self._apply_manual_roi_to_processor()
        self._refresh_display()

    def _confirm_ring_selection(self, event: Any = None) -> None:
        if not self._ring_edit_active:
            return
        preview = self._ring_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("Ring selection cancelled")
            self._cancel_ring_selection()
            return
        self._manual_ring = dict(preview)
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_ring_btn"):
            self._ring_btn.config(text="Set Ring")
        self._ring_status_var.set(
            f"Manual Ring: Locked ({int(round(self._manual_ring['radius'] * 2.0))} px)"
        )
        self._status_var.set("Manual docked ring locked and applied to offset calculation")
        self._show_ring_center.set(True)
        self._apply_manual_ring_to_processor()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._refresh_display()

    def _nudge_roi(self, dx: int, dy: int, event: Any = None) -> None:
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        step = 10.0 if (event is not None and (event.state & 0x0001)) else 2.0
        h, w = self._current_image.shape[:2]
        radius = self._roi_preview["radius"]
        self._roi_preview["center_x"] = float(
            np.clip(self._roi_preview["center_x"] + dx * step, radius, w - radius)
        )
        self._roi_preview["center_y"] = float(
            np.clip(self._roi_preview["center_y"] + dy * step, radius, h - radius)
        )
        self._status_var.set(
            "ROI edit mode: arrows nudge, drag move/resize, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _on_canvas_wheel(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_wheel(event)
            return
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._roi_preview["radius"] + direction * step
        self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "ROI edit mode: wheel resizes, arrows nudge, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _cancel_active_selection(self, event: Any = None) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._cancel_ruler_calibration(event)
            return
        if self._ring_edit_active:
            self._cancel_ring_selection(event)
            return
        self._cancel_roi_selection(event)

    def _canvas_to_image_point(
        self, canvas_x: float, canvas_y: float
    ) -> Optional[Tuple[float, float]]:
        if self._current_image is None:
            return None
        ox, oy = self._display_origin
        dw, dh = self._display_size
        if dw <= 0 or dh <= 0:
            return None
        if not (ox <= canvas_x <= ox + dw and oy <= canvas_y <= oy + dh):
            return None
        x = (canvas_x - ox) / max(self._display_scale, 1e-6)
        y = (canvas_y - oy) / max(self._display_scale, 1e-6)
        h, w = self._current_image.shape[:2]
        return (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))

    def _active_manual_roi(self) -> Optional[Dict[str, float]]:
        if self._manual_roi is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_roi.get("frame_width", w))) != w
            or int(round(self._manual_roi.get("frame_height", h))) != h
        ):
            return None
        return self._manual_roi

    def _active_manual_ring(self) -> Optional[Dict[str, float]]:
        if self._manual_ring is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_ring.get("frame_width", w))) != w
            or int(round(self._manual_ring.get("frame_height", h))) != h
        ):
            return None
        return self._manual_ring

    def _apply_manual_ring_policy(self, result: Any) -> Any:
        """Only allow ring data when a manual ring has been confirmed."""
        if result is None:
            return None
        ring = self._active_manual_ring()
        if ring is None:
            setattr(result, "ring_status", "ring_absent")
            setattr(result, "ring_center", None)
            setattr(result, "ring_radius", None)
            setattr(result, "ring_contour", None)
            setattr(result, "ring_dot_count", 0)
            setattr(result, "ring_confidence", 0.0)
            if "ring" in str(getattr(result, "corneal_reference_source", "")):
                setattr(result, "corneal_reference_source", "limbus")
            return result

        center = (float(ring["center_x"]), float(ring["center_y"]))
        radius = float(ring["radius"])
        setattr(result, "ring_status", "ring_present")
        setattr(result, "ring_center", center)
        setattr(result, "ring_radius", radius)
        setattr(result, "ring_contour", None)
        setattr(result, "ring_dot_count", int(round(ring.get("dot_count", 12))))
        setattr(result, "ring_confidence", 1.0)
        setattr(result, "corneal_reference_source", "manual_ring")
        return result

    def _apply_manual_roi_to_processor(self) -> None:
        roi = self._active_manual_roi()
        if self._opt_processor is None:
            return
        if roi is None:
            self._opt_processor.clear_manual_roi()
            return
        self._opt_processor.set_manual_roi(
            center_x=roi["center_x"],
            center_y=roi["center_y"],
            radius=roi["radius"],
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _apply_manual_ring_to_processor(self) -> None:
        ring = self._active_manual_ring()
        if self._opt_processor is None:
            return
        if ring is None:
            self._opt_processor.clear_manual_ring()
            return
        self._opt_processor.set_manual_ring(
            center_x=ring["center_x"],
            center_y=ring["center_y"],
            radius=ring["radius"],
            dot_count=int(round(ring.get("dot_count", 12))),
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _suggest_manual_ring_preview(self) -> Dict[str, float]:
        if self._current_result is not None:
            ring_status = getattr(self._current_result, "ring_status", "unknown")
            ring_center = getattr(self._current_result, "ring_center", None)
            ring_radius = getattr(self._current_result, "ring_radius", None)
            if (
                ring_status == "ring_present"
                and ring_center is not None
                and ring_radius is not None
            ):
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(ring_center[0]),
                    "center_y": float(ring_center[1]),
                    "radius": float(ring_radius),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(getattr(self._current_result, "ring_dot_count", 12)),
                }
        if self._opt_processor is not None and self._current_image is not None:
            suggestion = self._opt_processor.get_manual_ring(self._current_image.shape)
            if suggestion is not None:
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(suggestion["center_x"]),
                    "center_y": float(suggestion["center_y"]),
                    "radius": float(suggestion["radius"]),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(suggestion.get("dot_count", 12)),
                }
        h, w = self._current_image.shape[:2]
        roi = self._active_manual_roi()
        if roi is not None:
            center_x = float(roi["center_x"])
            center_y = float(roi["center_y"])
            radius = max(12.0, float(roi["radius"]) * 0.92)
        else:
            center_x = w / 2.0
            center_y = h / 2.0
            if self._current_result is not None and hasattr(self._current_result, "calibration") and self._current_result.calibration.calibrated:
                px_per_mm = self._current_result.calibration.px_per_mm
                radius = max(12.0, (14.1 * px_per_mm) / 2.0)
            else:
                radius = max(12.0, min(w, h) * 0.42)
        return {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "frame_width": float(w),
            "frame_height": float(h),
            "dot_count": 12.0,
        }

    def _handle_ring_canvas_press(self, event: Any) -> None:
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        if self._ring_preview is None:
            self._ring_preview = self._suggest_manual_ring_preview()
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        radius = self._ring_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.15)
        if abs(distance - radius) <= rim_threshold:
            self._ring_drag_mode = "resize"
        elif distance < radius:
            self._ring_drag_mode = "move"
            self._ring_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._ring_drag_mode = "resize"
            self._ring_preview["center_x"] = point[0]
            self._ring_preview["center_y"] = point[1]
            self._ring_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _handle_ring_canvas_drag(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_drag_mode is None
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return
        h, w = self._current_image.shape[:2]
        if self._ring_drag_mode == "move":
            radius = self._ring_preview["radius"]
            cx = point[0] - self._ring_drag_offset[0]
            cy = point[1] - self._ring_drag_offset[1]
            self._ring_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._ring_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._ring_preview["center_x"]
            cy = self._ring_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _handle_ring_canvas_release(self, event: Any) -> None:
        if not self._ring_edit_active:
            return
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._ring_preview is not None:
            self._status_var.set(
                "Ring ready. Match the red-dot circle, then press Enter to lock or Esc to cancel"
            )
        self._refresh_display()

    def _handle_ring_canvas_wheel(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._ring_preview["radius"] + direction * step
        self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "Ring edit mode: wheel resizes, drag move/resize, Enter lock, Esc cancel"
        )
        self._refresh_display()

    def _get_manual_roi_crop(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        roi = self._active_manual_roi()
        if roi is None:
            return None
        h, w = frame.shape[:2]
        cx = float(np.clip(roi["center_x"], 0, w - 1))
        cy = float(np.clip(roi["center_y"], 0, h - 1))
        radius = max(1.0, float(roi["radius"]))
        x0 = max(0, int(np.floor(cx - radius)))
        y0 = max(0, int(np.floor(cy - radius)))
        x1 = min(w, int(np.ceil(cx + radius)))
        y1 = min(h, int(np.ceil(cy + radius)))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return crop, float(x0), float(y0)

    def _on_canvas_resize(self, _event: Any) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(50, self._debounced_resize)

    def _debounced_resize(self) -> None:
        self._resize_after_id = None
        self._canvas_image_id = None
        self._refresh_display()

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


class SettingsMixin:
    def _toggle_grayscale(self, event=None) -> None:
        """Cycle grayscale mode: OFF → AUTO → FORCE → OFF.

        Called by the [G] keyboard shortcut and the toolbar button.
        """
        current = self._grayscale_mode_var.get()
        try:
            idx = _GRAYSCALE_CYCLE.index(current)
        except ValueError:
            idx = 0
        next_mode = _GRAYSCALE_CYCLE[(idx + 1) % len(_GRAYSCALE_CYCLE)]
        self._grayscale_mode_var.set(next_mode)
        self._on_grayscale_changed()

    def _on_grayscale_changed(self, *_args) -> None:
        """Apply grayscale mode change to the detector and update UI.

        Called when the settings dropdown or toolbar button changes.
        """
        mode = self._grayscale_mode_var.get()

        # Apply to detector
        if self._detector is not None:
            self._detector.set_grayscale_mode(mode)

        # Update toolbar button appearance
        label = _GRAYSCALE_LABELS.get(mode, "RGB")
        color = _GRAYSCALE_COLORS.get(mode, "#aaaaaa")
        if hasattr(self, "_gray_btn"):
            self._gray_btn.config(text=f"🔲 {label}")
        if hasattr(self, "_gray_indicator"):
            self._gray_indicator.config(
                text=f"  {label}  ",
                foreground=color,
            )

        # Update status
        mode_names = {
            "off": "RGB (original)",
            "auto": "Auto-detect",
            "force": "Forced grayscale",
        }
        self._status_var.set(f"Grayscale mode: {mode_names.get(mode, mode)}")

        # Refresh display immediately
        self._refresh_display()

    def _convert_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale for display when mode is active.

        When grayscale mode is FORCE, the displayed image becomes
        grayscale (like an IR camera) with coloured overlays on top.

        When mode is AUTO, converts only if the detector detected
        the input as grayscale.

        When mode is OFF, returns the original frame unchanged.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame.

        Returns
        -------
        np.ndarray
            Frame for display — 3-channel BGR uint8.
        """
        mode = self._grayscale_mode_var.get()

        if mode == "off":
            return frame

        if mode == "force":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if mode == "auto":
            # Convert only if detector applied grayscale processing
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return frame

    def _bind_live_setting_callbacks(self) -> None:
        callbacks = (
            (self._use_optimized_var, "pipeline"),
            (self._fp16_var, "engine"),
            (self._compile_var, "engine"),
            (self._resolution_var, "resolution"),
            (self._stride_var, "stride"),
            (self._target_fps_var, "display"),
            (self._roi_var, "roi"),
            (self._roi_cache_var, "roi"),
            (self._kalman_process_var, "tracking"),
            (self._kalman_measure_var, "tracking"),
            (self._calibration_mode_var, "calibration"),
            (self._fixed_scale_var, "calibration"),
            (self._corneal_ref_mm_var, "calibration"),
            (self._ring_ref_mm_var, "calibration"),
        )
        for var, reason in callbacks:
            var.trace_add(
                "write",
                lambda *_args, _reason=reason: self._schedule_live_settings_apply(
                    _reason
                ),
            )

    def _schedule_live_settings_apply(self, reason: str) -> None:
        if self._suspend_live_settings_apply:
            return
        self._pending_live_apply_reasons.add(reason)
        if self._settings_apply_after_id is not None:
            self.root.after_cancel(self._settings_apply_after_id)
        self._settings_apply_after_id = self.root.after(
            180,
            self._apply_live_settings,
        )

    def _apply_live_settings(self) -> None:
        self._settings_apply_after_id = None
        reasons = set(self._pending_live_apply_reasons)
        self._pending_live_apply_reasons.clear()
        if not reasons:
            return
        if self._restart_in_progress:
            self._pending_live_apply_reasons.update(reasons)
            self._settings_apply_after_id = self.root.after(250, self._apply_live_settings)
            return

        self._res_display.set(str(int(self._resolution_var.get())))
        self._kp_display.set(f"{float(self._kalman_process_var.get()):.3f}")
        self._km_display.set(f"{float(self._kalman_measure_var.get()):.3f}")

        runtime_restart_reasons = {"pipeline", "engine", "resolution", "stride", "roi", "tracking"}
        restart_required = bool(reasons.intersection(runtime_restart_reasons))

        if hasattr(self.cfg, "video"):
            self.cfg.video.kalman_process_noise = float(self._kalman_process_var.get())
            self.cfg.video.kalman_measurement_noise = float(
                self._kalman_measure_var.get()
            )

        # Apply calibration settings
        if "calibration" in reasons or not restart_required:
            cal_mode = self._calibration_mode_var.get()
            manual_px = float(self._fixed_scale_var.get())
            corneal_mm = float(self._corneal_ref_mm_var.get())
            ring_mm = float(self._ring_ref_mm_var.get())
            if hasattr(self.cfg, "calibration"):
                self.cfg.calibration.mode = cal_mode
                self.cfg.calibration.manual_px_per_mm = manual_px
                self.cfg.calibration.corneal_diameter_mm = corneal_mm
                self.cfg.calibration.suction_ring_diameter_mm = ring_mm
            if self._detector is not None:
                self._detector.set_calibration_mode(
                    mode=cal_mode,
                    manual_px_per_mm=manual_px,
                    corneal_diameter_mm=corneal_mm,
                    ring_diameter_mm=ring_mm,
                )

        if self._tracker is not None and not self._video_running:
            self._tracker = EyeKalmanTracker(config=self.cfg)

        if self._opt_processor is not None and not restart_required:
            try:
                self._opt_processor.update_runtime_settings(
                    enable_auto_roi=self._roi_var.get(),
                    roi_cache_ttl=self._roi_cache_var.get(),
                    process_noise=self._kalman_process_var.get(),
                    measurement_noise=self._kalman_measure_var.get(),
                )
                self._apply_manual_roi_to_processor()
            except Exception as exc:
                self.logger.warning("Live optimized settings apply failed: %s", exc)

        if restart_required and self._video_running and self._active_source is not None:
            self._restart_active_stream(
                f"Applied {' / '.join(sorted(reasons))} settings",
                rebuild_engine=bool(
                    reasons.intersection({"pipeline", "engine", "resolution"})
                ),
            )
            return

        if (
            reasons.intersection({"pipeline", "engine", "resolution"})
            and not self._video_running
        ):
            self._invalidate_fast_engine()
            self._get_fast_engine()

        self._status_var.set(
            "Settings applied live: "
            + ", ".join(r.replace("_", " ").title() for r in sorted(reasons))
        )
        self._refresh_display()

        # When calibration changes, re-compute mm values on the
        # currently displayed result so the measurement panel updates
        # immediately (without requiring a new image load / detection).
        if "calibration" in reasons and self._current_result is not None:
            res = self._current_result
            new_cal = self._detector._calibration if self._detector is not None else None
            if new_cal is not None and new_cal.calibrated:
                res.calibration = new_cal
            elif (
                new_cal is not None
                and not new_cal.calibrated
                and getattr(res, "limbus", None) is not None
                and getattr(res.limbus, "detected", False)
                and getattr(res.limbus, "ellipse", None) is not None
            ):
                # ANATOMICAL_ANCHOR after reset: _current_best() returns
                # uncalibrated because _ema_px_per_mm is None.  Compute
                # the tautological calibration from the current result's
                # pixel geometry so the panel shows correct values.
                ep = res.limbus.ellipse
                corneal_mm = float(self._corneal_ref_mm_var.get())
                dia_px = ep.semi_major * 2.0
                if dia_px > 10:
                    res.calibration = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=dia_px / corneal_mm,
                        mm_per_px=corneal_mm / dia_px,
                        source="anatomical_on_switch",
                        method="anatomical",
                        corneal_diameter_assumed_mm=corneal_mm,
                    )
            # Clear stale pre-computed mm attributes that were set by
            # _add_mm_values / evaluate_clinical_wtw during the original
            # detection.  These would otherwise be read by the display
            # code and show values computed with the OLD calibration.
            # (LimbusDetection/PupilDetection are dataclasses, so we
            # reset fields to None rather than delattr which does not
            # truly remove dataclass fields.)
            for target in (
                getattr(res, "limbus", None),
                getattr(res, "pupil", None),
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
            self._update_measurements(res)

    def _restart_active_stream(
        self, reason: str, rebuild_engine: bool = False
    ) -> None:
        if self._restart_in_progress:
            return
        source = self._active_source
        if source is None:
            return
        self._restart_in_progress = True
        self._status_var.set(f"{reason} - restarting stream...")
        try:
            if self._recorder.is_recording:
                self._stop_recording()
            if rebuild_engine:
                self._invalidate_fast_engine()
            self._stop_video()
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during stop: %s", exc)
            self._restart_in_progress = False
            self._report_runtime_issue("Restart failed while stopping the active stream")
            return

        self.root.after(120, lambda src=source: self._complete_stream_restart(src))

    def _complete_stream_restart(self, source: Any) -> None:
        try:
            self._start_video(source)
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during start: %s", exc)
            self._report_runtime_issue("Restart failed while starting the active stream")
        finally:
            self._restart_in_progress = False
            if self._pending_live_apply_reasons:
                self._schedule_live_settings_apply("restart")

    def _find_model_path(self) -> Optional[str]:
        candidates: List[str] = []
        if isinstance(self.cfg, dict):
            cfg_path = self.cfg.get("model_path")
            if cfg_path:
                candidates.append(str(cfg_path))
        if self._detector is not None:
            eng = getattr(self._detector, "ml_engine", None)
            if eng is not None:
                for attr in ("model_path", "_model_path"):
                    p = getattr(eng, attr, None)
                    if p:
                        candidates.append(str(p))
        candidates += [
            "models/best_model.pth",
            "model/best_model.pth",
            "pupil_tracking/models/best_model.pth",
        ]
        for c in candidates:
            if Path(c).is_file():
                found_path = str(Path(c).resolve())
                self.logger.info(f"Loaded model from: {found_path}")
                return found_path
        return None

    def _get_fast_engine(self) -> Optional[Any]:
        if not _FAST_PIPELINE_AVAILABLE:
            return None
        if self._fast_engine is not None:
            return self._fast_engine
        model_path = self._find_model_path()
        if model_path is None:
            self.logger.warning(
                "Cannot locate model file — optimised pipeline disabled"
            )
            return None
        try:
            self._fast_engine = FastInference(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                use_half=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                reflection_removal=True,
                suction_ring_removal=True,
            )
            self.logger.info(
                "FastInference ready (%s)",
                self._fast_engine.device,
            )
            self._engine_status_var.set(f"Engine: ready ({self._fast_engine.device})")
            return self._fast_engine
        except Exception as exc:
            self.logger.error("FastInference init failed: %s", exc)
            self._engine_status_var.set(f"Engine: error — {exc}")
            return None

    def _invalidate_fast_engine(self) -> None:
        if self._fast_engine is None:
            return
        try:
            import torch

            if (
                hasattr(self._fast_engine, "model")
                and self._fast_engine.model is not None
            ):
                del self._fast_engine.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._fast_engine = None
        self._engine_status_var.set("Engine: invalidated (will rebuild on next use)")

    def _rebuild_engine_ui(self) -> None:
        self._invalidate_fast_engine()
        engine = self._get_fast_engine()
        if engine is not None:
            engine.warmup()
            msg = (
                f"Engine: rebuilt ({engine.device}, "
                f"res={self._resolution_var.get()}, "
                f"fp16={self._fp16_var.get()})"
            )
            self._engine_status_var.set(msg)
            self._status_var.set(msg)
        else:
            self._engine_status_var.set("Engine: rebuild FAILED")
            self._status_var.set("Engine rebuild failed — check model path and logs")

    def _apply_performance_preset(self) -> None:
        preset = self._performance_preset_var.get()
        base_res = int(self._runtime_profile.recommended_resolution)
        base_fps = float(self._runtime_profile.recommended_target_fps)
        self._suspend_live_settings_apply = True
        try:
            if preset == "max_accuracy":
                self._resolution_var.set(
                    384 if self._runtime_profile.has_cuda else max(320, base_res)
                )
                self._target_fps_var.set(20.0 if self._runtime_profile.has_cuda else min(18.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            elif preset == "low_latency":
                self._resolution_var.set(256 if self._runtime_profile.has_cuda else 288)
                self._target_fps_var.set(30.0 if self._runtime_profile.has_cuda else max(20.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            else:
                self._resolution_var.set(base_res)
                self._target_fps_var.set(base_fps)
                self._stride_var.set(1)
                self._roi_var.set(True)
        finally:
            self._suspend_live_settings_apply = False
        self._res_display.set(str(self._resolution_var.get()))
        self._pending_live_apply_reasons.update({"resolution", "display", "stride", "roi"})
        self._schedule_live_settings_apply("preset")
        self._status_var.set(f"Preset: {preset.replace('_', ' ').title()} applied")

    def _get_display_interval(self) -> float:
        preset = self._performance_preset_var.get()
        preset_cap = {
            "max_accuracy": 20.0,
            "balanced": 28.0,
            "low_latency": 36.0,
        }.get(preset, _DISPLAY_FPS_CAP)
        target = float(self._target_fps_var.get() or _DISPLAY_FPS_CAP)
        cap = max(5.0, min(_DISPLAY_FPS_CAP, preset_cap, target))
        return 1.0 / cap

    def _derive_tracking_state(
        self, result: Any, stats: Optional[Dict[str, Any]] = None
    ) -> str:
        conf = float(getattr(result, "overall_confidence", 0.0) or 0.0)
        has_both = bool(getattr(result, "has_both", False))
        stats = stats or {}
        stale = int(stats.get("stale_frames", 0))
        dropped = int(stats.get("dropped_frames", 0))
        recent_latency = float(stats.get("latency_avg_ms", 0.0) or 0.0)

        if not has_both:
            return "No Detection"
        if stale > 10 or dropped > 10 or recent_latency > 250.0:
            return "Tracking Degraded"
        if conf >= 0.75:
            return "Tracking Stable"
        if conf >= 0.35:
            return "Tracking Acquiring"
        return "Tracking Degraded"

    def _set_summary_tracking_state(self, tracking_text: str) -> None:
        self._summary_tracking_var.set(tracking_text)
        tracking_color = {
            "Tracking Stable": self._colors.SURGICAL,
            "Tracking Acquiring": self._colors.CLINICAL,
            "Tracking Degraded": self._colors.RESEARCH,
            "No Detection": self._colors.INSUFFICIENT,
            "Ready": self._colors.ACCENT,
            "Waiting": self._colors.FG_SECONDARY,
        }.get(tracking_text, self._colors.FG_PRIMARY)
        self._summary_tracking_label.config(foreground=tracking_color)

    def _toggle_pause(self) -> None:
        if not self._video_running:
            return
        self._video_paused = not self._video_paused
        if self._video_paused:
            self._pause_btn.config(text="▶ Resume")
            self._status_var.set("Paused")
        else:
            self._pause_btn.config(text="⏸ Pause")
            self._status_var.set("Resumed")

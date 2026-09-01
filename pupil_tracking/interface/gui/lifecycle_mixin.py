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


class LifecycleMixin:
    def _init_settings_vars(self) -> None:
        self._use_optimized_var = tk.BooleanVar(value=True)
        self._fp16_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_fp16
        )
        self._compile_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_compile
        )

        self._resolution_var = tk.IntVar(
            value=self._runtime_profile.recommended_resolution
        )
        self._stride_var = tk.IntVar(value=1)
        self._target_fps_var = tk.DoubleVar(
            value=self._runtime_profile.recommended_target_fps
        )
        self._performance_preset_var = tk.StringVar(value="balanced")

        self._roi_var = tk.BooleanVar(value=True)
        self._roi_cache_var = tk.IntVar(value=5)
        self._roi_status_var = tk.StringVar(value="Manual ROI: Off")
        self._ring_status_var = tk.StringVar(value="Manual Ring: Off")
        self._kalman_process_var = tk.DoubleVar(value=0.03)
        self._kalman_measure_var = tk.DoubleVar(value=0.1)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 3 of 12 — Grayscale mode Tk variable
        # ══════════════════════════════════════════════════════════
        self._grayscale_mode_var = tk.StringVar(value="off")
        # ══════════════════════════════════════════════════════════

        self._pupil_fill_alpha_var = tk.IntVar(value=0)
        self._limbus_fill_alpha_var = tk.IntVar(value=0)

        # ── Modular Calibration Settings ──
        init_mode = getattr(self.cfg.calibration, "mode", "ANATOMICAL_ANCHOR") if hasattr(self.cfg, "calibration") else "ANATOMICAL_ANCHOR"
        self._calibration_mode_var = tk.StringVar(value=init_mode)
        init_manual_px = float(getattr(self.cfg.calibration, "manual_px_per_mm", 44.5) or 44.5) if hasattr(self.cfg, "calibration") else 44.5
        self._fixed_scale_var = tk.DoubleVar(value=init_manual_px)
        init_corneal = float(getattr(self.cfg.calibration, "corneal_diameter_mm", 12.0) or 12.0) if hasattr(self.cfg, "calibration") else 12.0
        self._corneal_ref_mm_var = tk.DoubleVar(value=init_corneal)
        init_ring = float(getattr(self.cfg.calibration, "suction_ring_diameter_mm", 9.4) or 9.4) if hasattr(self.cfg, "calibration") else 9.4
        self._ring_ref_mm_var = tk.DoubleVar(value=init_ring)

        # ── Interactive 2-Point Ruler Tool State ──
        self._ruler_calibration_active: bool = False
        self._ruler_points: list[tuple[float, float]] = []
        self._ruler_known_dist_mm_var = tk.DoubleVar(value=10.0)

    def _init_detector(self) -> None:
        self._status_var.set("Loading model…")
        self.root.update()

        try:
            self._tracker = EyeKalmanTracker(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init tracker: %s", exc)
            self._tracker = None

        try:
            self._corneal_calc = CornealCenterCalculator(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init corneal calc: %s", exc)
            self._corneal_calc = None

        try:
            # ══════════════════════════════════════════════════════
            # GRAYSCALE GUI 4 of 12 — Pass grayscale_mode to detector
            # ══════════════════════════════════════════════════════
            self._detector = UnifiedDetector(
                config=self.cfg,
                grayscale_mode=self._grayscale_mode_var.get(),
            )
            # ══════════════════════════════════════════════════════

            if self._detector.ml_engine.available:
                tag = "GPU" if _FAST_PIPELINE_AVAILABLE else "GPU (classic)"
                self._model_status_var.set(f"Model: Ready ({tag})")
            else:
                self._model_status_var.set("Model: Classical Only (ML unavailable)")

            self._status_var.set("Ready — Load an image or start camera")
        except Exception as exc:
            self.logger.error("Failed to init detector: %s", exc)
            self._detector = None
            self._model_status_var.set(f"Model: ERROR — {exc}")
            self._status_var.set(
                "Model loading failed — classical detection unavailable"
            )

    def _install_crash_guards(self) -> None:
        self.root.report_callback_exception = self._handle_tk_exception
        threading.excepthook = self._handle_thread_exception

    def _handle_tk_exception(self, exc_type, exc_value, exc_traceback) -> None:
        self.logger.exception(
            "Tk callback error",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        self._report_runtime_issue("UI callback error recovered")

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        self.logger.exception(
            "Worker thread error in %s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        self.root.after(
            0,
            lambda: self._report_runtime_issue("Background worker recovered"),
        )

    def _report_runtime_issue(self, message: str) -> None:
        try:
            self._status_var.set(message)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._stop_video()
        # ══════════════════════════════════════════════════════════
        # RECORDING — Cleanup recording on close
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()
        # ══════════════════════════════════════════════════════════
        if self._fast_engine is not None:
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
        if _FAST_PIPELINE_AVAILABLE and FastInference is not None:
            try:
                FastInference.reset_cache()
            except Exception:
                pass
        try:
            self.logger.close()
        except Exception:
            pass
        self.root.destroy()

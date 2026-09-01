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


class BuildersMixin:
    def _build_menu(self) -> None:
        c = self._colors
        menu_cfg = dict(
            bg=c.BG_SECONDARY,
            fg=c.FG_PRIMARY,
            activebackground=c.ACCENT_DIM,
            activeforeground=c.FG_PRIMARY,
            relief="flat",
            borderwidth=0,
        )
        menubar = tk.Menu(self.root, **menu_cfg)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Image…",
            command=self._open_image,
            accelerator="Ctrl+O",
        )
        file_menu.add_command(
            label="Open Video…",
            command=self._open_video,
            accelerator="Ctrl+V",
        )
        file_menu.add_separator()
        file_menu.add_command(label="Open Folder…", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results CSV…", command=self._export_csv)
        file_menu.add_command(label="Export Results JSON…", command=self._export_json)
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        # ══════════════════════════════════════════════════════════
        # RECORDING 3 of 8 — Recording menu options
        # ══════════════════════════════════════════════════════════
        file_menu.add_separator()
        self._recording_menu_var = tk.StringVar(value="Start Recording")
        file_menu.add_command(
            label="Start Recording…",
            command=self._start_recording,
            accelerator="Ctrl+R",
        )
        file_menu.add_command(
            label="Stop Recording",
            command=self._stop_recording,
            accelerator="Ctrl+Shift+R",
        )
        file_menu.add_separator()
        # ══════════════════════════════════════════════════════════
        file_menu.add_command(
            label="Exit", command=self._on_close, accelerator="Ctrl+Q"
        )

        camera_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Start Camera", command=self._start_camera)
        camera_menu.add_command(label="Stop Camera", command=self._stop_video)

        view_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="View", menu=view_menu)
        self._show_overlay = tk.BooleanVar(value=True)
        self._show_pupil = tk.BooleanVar(value=True)
        self._show_limbus = tk.BooleanVar(value=True)
        self._show_offset = tk.BooleanVar(value=True)
        self._show_centers = tk.BooleanVar(value=True)
        self._show_ring_center = tk.BooleanVar(value=False)
        self._show_measurements = tk.BooleanVar(value=False)
        self._show_debug_overlay = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Show Overlay",
            variable=self._show_overlay,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Pupil",
            variable=self._show_pupil,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Limbus",
            variable=self._show_limbus,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Offset Line",
            variable=self._show_offset,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Centers",
            variable=self._show_centers,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Ring Center",
            variable=self._show_ring_center,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show On-Image Measurements",
            variable=self._show_measurements,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Debug Overlay",
            variable=self._show_debug_overlay,
            command=self._refresh_display,
        )

        self.root.bind("<Control-o>", lambda _e: self._open_image())
        self.root.bind("<Control-v>", lambda _e: self._open_video())
        self.root.bind("<Control-q>", lambda _e: self._on_close())
        self.root.bind("<space>", lambda _e: self._toggle_pause())
        # ══════════════════════════════════════════════════════════
        # RECORDING 4 of 8 — Recording keyboard shortcuts
        # ══════════════════════════════════════════════════════════
        self.root.bind("<Control-r>", lambda _e: self._start_recording())
        self.root.bind("<Control-R>", lambda _e: self._toggle_recording())
        # ══════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 6 of 12 — [G] keyboard shortcut
        # ══════════════════════════════════════════════════════════
        self.root.bind("<g>", self._toggle_grayscale)
        self.root.bind("<G>", self._toggle_grayscale)
        self.root.bind("<Return>", self._confirm_active_selection)
        self.root.bind("<Escape>", self._cancel_active_selection)
        self.root.bind("<Left>", lambda e: self._nudge_roi(-1, 0, e))
        self.root.bind("<Right>", lambda e: self._nudge_roi(1, 0, e))
        self.root.bind("<Up>", lambda e: self._nudge_roi(0, -1, e))
        self.root.bind("<Down>", lambda e: self._nudge_roi(0, 1, e))

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, style="Primary.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 1))

        ttk.Button(toolbar, text="📂 Image", command=self._open_image).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="🎞 Video", command=self._open_video).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="📷 Camera", command=self._start_camera).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self._pause_btn = ttk.Button(
            toolbar,
            text="⏸ Pause",
            command=self._toggle_pause,
            state=tk.DISABLED,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="⏹ Stop", command=self._stop_video).pack(
            side=tk.LEFT, padx=2
        )

        # ══════════════════════════════════════════════════════════
        # RECORDING — Recording toolbar button
        # ══════════════════════════════════════════════════════════
        self._rec_btn = ttk.Button(
            toolbar,
            text="⏺ Start Rec",
            command=self._toggle_recording,
        )
        self._rec_btn.pack(side=tk.LEFT, padx=2)

        self._recording_indicator = ttk.Label(
            toolbar,
            text="  --:--  ",
            style="Quality.TLabel",
            foreground="#616161",
        )
        self._recording_indicator.pack(side=tk.LEFT, padx=(0, 4))
        # ══════════════════════════════════════════════════════════

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 7 of 12 — Grayscale toggle button in toolbar
        #
        # Button cycles: RGB → AUTO → GRAY → RGB
        # Indicator label shows current mode with colour coding
        # ══════════════════════════════════════════════════════════
        self._gray_btn = ttk.Button(
            toolbar,
            text="🔲 RGB",
            command=self._toggle_grayscale,
        )
        self._gray_btn.pack(side=tk.LEFT, padx=2)

        self._gray_indicator = ttk.Label(
            toolbar,
            text="  RGB  ",
            style="Quality.TLabel",
            foreground=_GRAYSCALE_COLORS["off"],
        )
        self._gray_indicator.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self._roi_btn = ttk.Button(
            toolbar,
            text="Set ROI",
            command=self._begin_roi_selection,
        )
        self._roi_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar,
            text="Clear ROI",
            command=self._clear_manual_roi,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            toolbar,
            textvariable=self._roi_status_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(4, 8))

        self._ring_btn = ttk.Button(
            toolbar,
            text="Set Ring",
            command=self._begin_ring_selection,
        )
        self._ring_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar,
            text="Clear Ring",
            command=self._clear_manual_ring,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            toolbar,
            textvariable=self._ring_status_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(4, 8))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="📏 Scale Wizard", command=self._open_calibration_wizard).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Export CSV", command=self._export_csv).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Snapshot", command=self._export_snapshot).pack(
            side=tk.LEFT, padx=2
        )


        self._quality_label = ttk.Label(
            toolbar,
            text="  NO IMAGE  ",
            style="Quality.TLabel",
            anchor="center",
        )
        self._quality_label.pack(side=tk.RIGHT, padx=10)

    def _build_main_area(self) -> None:
        c = self._colors
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Default split ratio: 7:3 (image/loading on the left, panels on the right)
        left_frame = ttk.Frame(main, style="Primary.TFrame")
        main.add(left_frame, weight=65)

        self._build_progress_frame(left_frame)

        self._canvas = tk.Canvas(
            left_frame,
            bg=c.CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_canvas_wheel)

        right_frame = ttk.Frame(main, width=500)
        main.add(right_frame, weight=35)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        meas_frame = ttk.Frame(notebook, padding=8)
        notebook.add(meas_frame, text="Measurements")
        self._build_measurements_panel(meas_frame)

        detail_frame = ttk.Frame(notebook, padding=8)
        notebook.add(detail_frame, text="Details")
        self._build_details_panel(detail_frame)

        settings_frame = ttk.Frame(notebook, padding=8)
        notebook.add(settings_frame, text="⚙ Settings")
        self._build_settings_panel(settings_frame)

        # Ensure the PanedWindow sash initial position is set once layout is realized.
        # Without this, some Tk themes/OS window managers keep the sash at a default
        # position until the user resizes the window.
        def _set_initial_sash() -> None:
            w = main.winfo_width()
            if w is None or w <= 1:
                self.root.after(150, _set_initial_sash)
                return
            main.sashpos(0, int(w * 0.70))

        self.root.after(0, _set_initial_sash)

    def _build_progress_frame(self, parent: ttk.Frame) -> None:
        self._progress_outer = ttk.LabelFrame(
            parent,
            text="Video Progress",
            padding=4,
        )
        self._progress_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=(2, 0))

        self._progress_bar = ttk.Progressbar(
            self._progress_outer, mode="determinate", maximum=100
        )
        self._progress_bar.pack(fill=tk.X, padx=4, pady=(2, 2))

        info_row = ttk.Frame(self._progress_outer)
        info_row.pack(fill=tk.X, padx=4, pady=(0, 2))

        self._progress_label_var = tk.StringVar(value="No video loaded")
        ttk.Label(
            info_row,
            textvariable=self._progress_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT)

        self._eta_label_var = tk.StringVar(value="")
        ttk.Label(
            info_row,
            textvariable=self._eta_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT)

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sn = ("Consolas", 9)
        sw = 24

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 8 of 12 — Grayscale section in Settings
        # ══════════════════════════════════════════════════════════
        g_lf = ttk.LabelFrame(sf, text="🔲 Grayscale Mode", padding=8)
        g_lf.pack(fill=tk.X, padx=4, pady=4)

        g_desc = ttk.Label(
            g_lf,
            text=(
                "Convert display to grayscale (like IR camera).\n"
                "Detection still works — overlays shown in colour.\n"
                "Press [G] to toggle quickly."
            ),
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        g_desc.pack(anchor=tk.W, pady=(0, 6))

        g_row = ttk.Frame(g_lf)
        g_row.pack(fill=tk.X, pady=2)
        ttk.Label(
            g_row,
            text="Mode:",
            font=sn,
            width=12,
        ).pack(side=tk.LEFT)

        for mode_val, mode_label, mode_desc in [
            ("off", "RGB (Original)", "Show original colour image"),
            ("auto", "Auto-Detect", "Grayscale only if input is grayscale"),
            ("force", "Force Grayscale", "Always show as grayscale (IR look)"),
        ]:
            rb = ttk.Radiobutton(
                g_lf,
                text=f"{mode_label}  —  {mode_desc}",
                variable=self._grayscale_mode_var,
                value=mode_val,
                command=self._on_grayscale_changed,
            )
            rb.pack(anchor=tk.W, padx=(20, 0), pady=1)

        self._gray_settings_status = tk.StringVar(value="Current: RGB")
        ttk.Label(
            g_lf,
            textvariable=self._gray_settings_status,
            font=sn,
            foreground=c.ACCENT,
        ).pack(anchor=tk.W, pady=(6, 0))
        # ══════════════════════════════════════════════════════════
        # END GRAYSCALE GUI 8
        # ══════════════════════════════════════════════════════════

        p_lf = ttk.LabelFrame(sf, text="Pipeline", padding=8)
        p_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            p_lf,
            text="Live Preset:",
            font=sn,
        ).pack(anchor=tk.W)
        for preset_value, preset_label in [
            ("max_accuracy", "Max Accuracy"),
            ("balanced", "Balanced"),
            ("low_latency", "Low Latency"),
        ]:
            ttk.Radiobutton(
                p_lf,
                text=preset_label,
                variable=self._performance_preset_var,
                value=preset_value,
                command=self._apply_performance_preset,
            ).pack(anchor=tk.W, padx=(20, 0), pady=1)

        ttk.Checkbutton(
            p_lf,
            text="Use Optimised Pipeline (when available)",
            variable=self._use_optimized_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="FP16 Half-Precision",
            variable=self._fp16_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="torch.compile (JIT)",
            variable=self._compile_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)

        avail = "✓ Available" if _FAST_PIPELINE_AVAILABLE else "✗ Not installed"
        ttk.Label(
            p_lf,
            text=f"Fast pipeline: {avail}",
            style="Muted.TLabel",
            foreground=(c.SURGICAL if _FAST_PIPELINE_AVAILABLE else c.INSUFFICIENT),
        ).pack(anchor=tk.W, pady=(4, 0))

        v_lf = ttk.LabelFrame(sf, text="Video Processing", padding=8)
        v_lf.pack(fill=tk.X, padx=4, pady=4)

        r_row = ttk.Frame(v_lf)
        r_row.pack(fill=tk.X, pady=2)
        ttk.Label(r_row, text="Inference Resolution:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._res_display = tk.StringVar(value=str(self._resolution_var.get()))
        ttk.Scale(
            r_row,
            from_=192,
            to=512,
            variable=self._resolution_var,
            command=lambda v: self._res_display.set(str(int(float(v)))),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(r_row, textvariable=self._res_display, font=sn, width=4).pack(
            side=tk.LEFT
        )

        s_row = ttk.Frame(v_lf)
        s_row.pack(fill=tk.X, pady=2)
        ttk.Label(s_row, text="Frame Stride:", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            s_row,
            from_=1,
            to=30,
            textvariable=self._stride_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(s_row, text="(1 = every frame)", font=sn).pack(side=tk.LEFT)

        f_row = ttk.Frame(v_lf)
        f_row.pack(fill=tk.X, pady=2)
        ttk.Label(f_row, text="Camera Target FPS:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            f_row,
            from_=5.0,
            to=60.0,
            textvariable=self._target_fps_var,
            width=5,
            font=sn,
            increment=5.0,
        ).pack(side=tk.LEFT, padx=4)

        t_lf = ttk.LabelFrame(sf, text="ROI & Tracking", padding=8)
        t_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Checkbutton(
            t_lf,
            text="Enable ROI Tracking",
            variable=self._roi_var,
        ).pack(anchor=tk.W)
        ttk.Label(
            t_lf,
            textvariable=self._roi_status_var,
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 4))

        rc_row = ttk.Frame(t_lf)
        rc_row.pack(fill=tk.X, pady=2)
        ttk.Label(rc_row, text="ROI Cache (frames):", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            rc_row,
            from_=1,
            to=30,
            textvariable=self._roi_cache_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        kp_row = ttk.Frame(t_lf)
        kp_row.pack(fill=tk.X, pady=2)
        ttk.Label(kp_row, text="Kalman Process Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._kp_display = tk.StringVar(value=f"{self._kalman_process_var.get():.3f}")
        ttk.Scale(
            kp_row,
            from_=0.001,
            to=0.5,
            variable=self._kalman_process_var,
            command=lambda v: self._kp_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(kp_row, textvariable=self._kp_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        km_row = ttk.Frame(t_lf)
        km_row.pack(fill=tk.X, pady=2)
        ttk.Label(km_row, text="Kalman Measure Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._km_display = tk.StringVar(value=f"{self._kalman_measure_var.get():.3f}")
        ttk.Scale(
            km_row,
            from_=0.01,
            to=1.0,
            variable=self._kalman_measure_var,
            command=lambda v: self._km_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(km_row, textvariable=self._km_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        cal_lf = ttk.LabelFrame(sf, text="Calibration & Physical Units", padding=8)
        cal_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            cal_lf,
            text="Select physical scale derivation method:",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))

        for mode_val, mode_text in [
            ("ANATOMICAL_ANCHOR", "Anatomical Anchor (Cornea ≈ 12.0 mm)"),
            ("FIXED_PIXEL_SCALE", "Fixed Pixel Scale (Manual / External Target)"),
            ("RING_REFLECTION", "Ring Reflection (Purkinje / Placido Ring)"),
        ]:
            ttk.Radiobutton(
                cal_lf,
                text=mode_text,
                variable=self._calibration_mode_var,
                value=mode_val,
                command=lambda: self._schedule_live_settings_apply("calibration"),
            ).pack(anchor=tk.W, padx=(8, 0), pady=1)

        px_mm_row = ttk.Frame(cal_lf)
        px_mm_row.pack(fill=tk.X, pady=2)
        ttk.Label(px_mm_row, text="Fixed Scale (px/mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            px_mm_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        cornea_row = ttk.Frame(cal_lf)
        cornea_row.pack(fill=tk.X, pady=2)
        ttk.Label(cornea_row, text="Assumed Cornea (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            cornea_row,
            from_=8.0,
            to=15.0,
            increment=0.1,
            textvariable=self._corneal_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ring_row = ttk.Frame(cal_lf)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ref Ring Dia (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=20.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            cal_lf,
            text="⚡ Open Calibration & WTW Wizard…",
            command=self._open_calibration_wizard,
        ).pack(anchor=tk.W, pady=(6, 2))

        fill_lf = ttk.LabelFrame(sf, text="Circle Fill Shading", padding=8)

        fill_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            fill_lf,
            text="Fill the pupil (green) and limbus (blue) circles.\n0 = no fill, 100 = fully coloured.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        for label_text, var, disp_attr in [
            ("Pupil (green) %:", self._pupil_fill_alpha_var, "_pupil_fill_display"),
            ("Limbus (blue) %:", self._limbus_fill_alpha_var, "_limbus_fill_display"),
        ]:
            disp_var = tk.StringVar(value="0")
            setattr(self, disp_attr, disp_var)
            row = ttk.Frame(fill_lf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text, font=sn, width=sw).pack(side=tk.LEFT)
            _v = var  # capture for lambda
            _d = disp_var
            ttk.Scale(
                row,
                from_=0,
                to=100,
                variable=_v,
                command=lambda v, d=_d: (
                    d.set(str(int(float(v)))),
                    self._refresh_display(),
                ),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Label(row, textvariable=disp_var, font=sn, width=4).pack(side=tk.LEFT)

        a_lf = ttk.LabelFrame(sf, text="Actions", padding=8)
        a_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(
            a_lf,
            text="Rebuild Inference Engine",
            command=self._rebuild_engine_ui,
        ).pack(anchor=tk.W)

        self._engine_status_var = tk.StringVar(value="Engine: not initialised")
        ttk.Label(
            a_lf,
            textvariable=self._engine_status_var,
            font=sn,
        ).pack(anchor=tk.W, pady=(4, 0))

        note = (
            "Most settings apply live. Pipeline and resolution\n"
            "changes restart the active stream automatically\n"
            "when needed. Use 'Rebuild Inference Engine'\n"
            "to refresh the fast path immediately. Press [G]\n"
            "to toggle grayscale mode at any time."
        )
        ttk.Label(
            sf,
            text=note,
            style="Tiny.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))

    def _build_measurements_panel(self, parent: ttk.Frame) -> None:
        c = self._colors

        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(scroll_window, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def add_card(
            pf,
            title: str,
            style_name: str,
            row: int,
            column: int,
            columnspan: int = 1,
        ) -> ttk.Frame:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=6)
            card.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            ttk.Label(
                card,
                text=title,
                style=style_name,
            ).pack(anchor=tk.W, pady=(0, 4))
            body = ttk.Frame(card, style="MetricCard.TFrame")
            body.pack(fill=tk.X)
            body.columnconfigure(1, weight=1)
            return body

        def add_summary_card(pf, title: str, column: int) -> Tuple[tk.StringVar, ttk.Label]:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=10)
            card.grid(row=0, column=column, sticky="nsew", padx=3, pady=2)
            card.pack_propagate(False)
            ttk.Label(card, text=title, style="CardKey.TLabel").pack(anchor=tk.W)
            var = tk.StringVar(value="---")
            label = ttk.Label(
                card,
                textvariable=var,
                style="CardValue.TLabel",
                width=18,
                anchor="w",
            )
            label.pack(anchor=tk.W, pady=(4, 0))
            return var, label

        def add_row(pf, label_text, width=16):
            row = ttk.Frame(pf, style="MetricCard.TFrame")
            row.pack(fill=tk.X, pady=1)
            row.columnconfigure(1, weight=1)
            ttk.Label(
                row,
                text=label_text,
                style="CardKey.TLabel",
                width=width,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value="---")
            ttk.Label(
                row,
                textvariable=var,
                style="CardValueSmall.TLabel",
                anchor="e",
                justify=tk.RIGHT,
            ).grid(row=0, column=1, sticky="ew")
            return var

        summary_outer = ttk.Frame(scroll_frame)
        summary_outer.pack(fill=tk.X, pady=(8, 6))
        summary_outer.columnconfigure(0, weight=1)
        summary_outer.columnconfigure(1, weight=1)
        summary_outer.columnconfigure(2, weight=1)
        summary_outer.columnconfigure(3, weight=1)
        self._summary_quality_var, self._summary_quality_label = add_summary_card(
            summary_outer, "Quality", 0
        )
        self._summary_tracking_var, self._summary_tracking_label = add_summary_card(
            summary_outer, "Tracking", 1
        )
        self._summary_latency_var, self._summary_latency_label = add_summary_card(
            summary_outer, "Latency", 2
        )
        self._summary_pipeline_var, self._summary_pipeline_label = add_summary_card(
            summary_outer, "Pipeline", 3
        )

        cards_outer = ttk.Frame(scroll_frame, style="Primary.TFrame")
        cards_outer.pack(fill=tk.BOTH, expand=True)
        cards_outer.columnconfigure(0, weight=1, uniform="measurement_cards")
        cards_outer.columnconfigure(1, weight=1, uniform="measurement_cards")

        pupil_frame = add_card(cards_outer, "PUPIL", "PupilHeader.TLabel", 0, 0)
        self._pv: Dict[str, tk.StringVar] = {}
        self._pv["center"] = add_row(pupil_frame, "Center:")
        self._pv["diameter_px"] = add_row(pupil_frame, "Diameter (px):")
        self._pv["diameter_mm"] = add_row(pupil_frame, "Diameter (mm):")
        self._pv["semi_major"] = add_row(pupil_frame, "Semi-Major (px):")
        self._pv["semi_major_mm"] = add_row(pupil_frame, "Semi-Major (mm):")
        self._pv["semi_minor"] = add_row(pupil_frame, "Semi-Minor (px):")
        self._pv["semi_minor_mm"] = add_row(pupil_frame, "Semi-Minor (mm):")
        self._pv["angle"] = add_row(pupil_frame, "Angle:")
        self._pv["fit_type"] = add_row(pupil_frame, "Fit Type:")
        self._pv["confidence"] = add_row(pupil_frame, "Confidence:")
        self._pv["quality"] = add_row(pupil_frame, "Quality:")

        limbus_frame = add_card(cards_outer, "LIMBUS", "LimbusHeader.TLabel", 0, 1)
        self._lv: Dict[str, tk.StringVar] = {}
        self._lv["center"] = add_row(limbus_frame, "Center:")
        self._lv["diameter_px"] = add_row(limbus_frame, "Diameter (px):")
        self._lv["diameter_mm"] = add_row(limbus_frame, "Diameter (mm):")
        self._lv["semi_major"] = add_row(limbus_frame, "Semi-Major (px):")
        self._lv["semi_major_mm"] = add_row(limbus_frame, "Semi-Major (mm):")
        self._lv["semi_minor"] = add_row(limbus_frame, "Semi-Minor (px):")
        self._lv["semi_minor_mm"] = add_row(limbus_frame, "Semi-Minor (mm):")
        self._lv["angle"] = add_row(limbus_frame, "Angle:")
        self._lv["fit_type"] = add_row(limbus_frame, "Fit Type:")
        self._lv["confidence"] = add_row(limbus_frame, "Confidence:")
        self._lv["quality"] = add_row(limbus_frame, "Quality:")

        offset_frame = add_card(
            cards_outer, "CORNEAL CENTRE & OFFSET", "OffsetHeader.TLabel", 1, 0
        )
        self._ov: Dict[str, tk.StringVar] = {}
        self._ov["corneal_center"] = add_row(offset_frame, "Corneal Centre:")
        self._ov["corneal_reference"] = add_row(offset_frame, "Reference:")
        self._ov["ring_center"] = add_row(offset_frame, "Ring Centre:")
        self._ov["ring_diameter_px"] = add_row(offset_frame, "Ring Dia (px):")
        self._ov["ring_diameter_mm"] = add_row(offset_frame, "Ring Dia (mm):")
        self._ov["offset_px"] = add_row(offset_frame, "Offset (px):")
        self._ov["offset_mm"] = add_row(offset_frame, "Offset (mm):")
        self._ov["offset_vec_px"] = add_row(offset_frame, "Offset dX,dY px:")
        self._ov["offset_vec_mm"] = add_row(offset_frame, "Offset dX,dY mm:")
        self._ov["offset_angle"] = add_row(offset_frame, "Offset Angle:")
        self._ov["pupil_limbus_ratio"] = add_row(offset_frame, "Pupil/Limbus:")

        calib_frame = add_card(cards_outer, "CALIBRATION", "CalibHeader.TLabel", 1, 1)
        self._cv_vars: Dict[str, tk.StringVar] = {}
        self._cv_vars["source"] = add_row(calib_frame, "Source:")
        self._cv_vars["scale_px"] = add_row(calib_frame, "px/mm:")
        self._cv_vars["scale_mm"] = add_row(calib_frame, "mm/px:")
        self._cv_vars["reference"] = add_row(calib_frame, "Reference:")

        wtw_frame = add_card(
            cards_outer, "CORNEAL DIMENSIONS (WTW)", "LimbusHeader.TLabel", 2, 0, 2
        )
        self._wtw_vars: Dict[str, tk.StringVar] = {}
        self._wtw_vars["horizontal"] = add_row(wtw_frame, "Horizontal WTW:")
        self._wtw_vars["vertical"] = add_row(wtw_frame, "Vertical WTW:")
        self._wtw_vars["mean"] = add_row(wtw_frame, "Mean WTW:")

        iris_frame = add_card(
            cards_outer, "CYCLOTORSION / IRIS", "OffsetHeader.TLabel", 2, 1
        )
        self._iris_vars: Dict[str, tk.StringVar] = {}
        self._iris_vars["status"] = add_row(iris_frame, "Status:")
        self._iris_vars["feature_count"] = add_row(iris_frame, "Features:")
        self._iris_vars["angular_coverage"] = add_row(iris_frame, "Coverage:")
        self._iris_vars["rotation_angle"] = add_row(iris_frame, "Rotation Angle:")
        self._iris_vars["confidence"] = add_row(iris_frame, "Confidence:")
        self._iris_vars["evidence"] = add_row(iris_frame, "Evidence:")

        proc_frame = add_card(cards_outer, "PROCESSING", "ProcHeader.TLabel", 3, 0, 2)
        self._proc_time_var = add_row(proc_frame, "Proc. Time:")
        self._latency_var = add_row(proc_frame, "Latency:")
        self._latency_avg_var = add_row(proc_frame, "Latency Avg:")
        self._drop_var = add_row(proc_frame, "Dropped/Stale:")
        self._tracking_state_var = add_row(proc_frame, "Tracking:")
        self._fps_var = add_row(proc_frame, "FPS:")
        self._frame_var = add_row(proc_frame, "Frame:")
        self._image_size_var = add_row(proc_frame, "Image Size:")
        self._pipeline_var = add_row(proc_frame, "Pipeline:")

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 9 of 12 — Grayscale info in measurements
        # ══════════════════════════════════════════════════════════
        self._gray_mode_var_display = add_row(proc_frame, "Grayscale:")

    def _build_details_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        self._details_text = tk.Text(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg=c.BG_INPUT,
            fg=c.FG_PRIMARY,
            state=tk.DISABLED,
            height=30,
            insertbackground=c.FG_PRIMARY,
            selectbackground=c.ACCENT_DIM,
            selectforeground=c.FG_PRIMARY,
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self._details_text.yview,
        )
        self._details_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._details_text.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self) -> None:
        status = ttk.Frame(self.root, style="Primary.TFrame")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status,
            textvariable=self._status_var,
            style="Status.TLabel",
        ).pack(side=tk.LEFT, padx=5)

        self._model_status_var = tk.StringVar(value="Model: Loading…")
        ttk.Label(
            status,
            textvariable=self._model_status_var,
            style="Status.TLabel",
        ).pack(side=tk.RIGHT, padx=5)

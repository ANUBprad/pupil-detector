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
from pupil_tracking.iris.detect import detect_iris_features
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


class MediaMixin:
    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpeg *.jpg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_video()
        self._load_and_detect_image(path)

    def _open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self._stop_video()
        images = (
            sorted(Path(folder).glob("*.jpeg"))
            + sorted(Path(folder).glob("*.jpg"))
            + sorted(Path(folder).glob("*.png"))
        )
        if not images:
            messagebox.showwarning("No Images", f"No images found in {folder}")
            return
        self._results_history.clear()
        for i, img_path in enumerate(images):
            self._status_var.set(f"Processing {i + 1}/{len(images)}: {img_path.name}")
            self.root.update()
            self._load_and_detect_image(str(img_path))
        self._status_var.set(f"Processed {len(images)} images")

    def _load_and_detect_image(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {path}")
            return
        self._current_image = image
        self._frame_count += 1
        if self._detector is None:
            self._status_var.set("Detector not ready — showing raw image")
            self._current_result = None
            self._refresh_display()
            return
        self._sync_calibration_to_detector()
        self._status_var.set(f"Detecting: {Path(path).name}…")
        self.root.update()
        result = self._detector.detect(
            image, frame_number=self._frame_count, source=path
        )
        result = self._apply_manual_ring_policy(result)
        result = self._run_iris_detection(image, result)
        self._current_result = result
        self._results_history.append(result.to_dict())
        self._update_measurements(result)
        self._refresh_display()
        self._status_var.set(
            f"{Path(path).name} — {result.overall_quality.value} "
            f"({result.overall_confidence:.3f}) — "
            f"{result.metadata.processing_time_ms:.0f} ms"
        )

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_video()
        self._start_video(path)

    def _start_video(self, source: Any) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {source}")
            return
        self._active_source = source
        self._video_cap = cap
        self._video_running = True
        self._video_paused = False
        self._using_optimized_camera = False
        self._camera_mode = isinstance(source, int)
        if self._tracker is not None:
            self._tracker.reset()
        self._results_history.clear()
        self._frame_count = 0
        self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_start_time = time.monotonic()
        self._last_display_update = 0.0
        if self._video_total_frames > 0:
            self._progress_bar.config(
                mode="determinate",
                maximum=self._video_total_frames,
            )
        else:
            self._progress_bar.config(mode="indeterminate", maximum=100)
        self._progress_bar["value"] = 0
        self._progress_label_var.set("Starting…")
        self._eta_label_var.set("")
        self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
        src_name = "Camera" if isinstance(source, int) else Path(str(source)).name
        use_opt = self._use_optimized_var.get() and _FAST_PIPELINE_AVAILABLE
        engine = self._get_fast_engine() if use_opt else None
        if engine is not None:
            self._start_video_optimized(engine, src_name)
        else:
            self._start_video_classic(src_name)

    def _start_video_classic(self, src_name: str) -> None:
        self._status_var.set(f"Playing (classic): {src_name}")
        self._pipeline_var.set("Classic")
        self._video_thread = threading.Thread(
            target=self._video_loop_classic,
            args=(src_name,),
            daemon=True,
            name="VideoLoopClassic",
        )
        self._video_thread.start()

    def _video_loop_classic(self, source_name: str) -> None:
        raw_frame_idx = 0
        consecutive_read_failures = 0
        max_read_failures = 10  # tolerate transient camera glitches
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                if not self._camera_mode:
                    self._video_running = False
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            f"Video complete: {len(self._results_history)} frames"
                        ),
                    )
                    self.root.after(0, self._on_video_complete)
                    break
                # Camera mode: tolerate transient failures
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            "Camera read failed repeatedly — stopping"
                        ),
                    )
                    self._video_running = False
                    break
                time.sleep(0.01)
                continue
            consecutive_read_failures = 0
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            self._frame_count += 1
            self._current_image = frame
            if self._detector is None:
                self.root.after(0, self._update_progress, raw_frame_idx, True)
                continue
            manual_crop = self._get_manual_roi_crop(frame)
            if manual_crop is not None:
                crop, roi_x, roi_y = manual_crop
                self._sync_calibration_to_detector()
                result = self._detector.detect_video_frame(
                    crop,
                    frame_number=self._frame_count,
                    roi_x=roi_x,
                    roi_y=roi_y,
                )
                result = self._apply_manual_ring_policy(result)
                result.metadata.source = source_name
            else:
                self._sync_calibration_to_detector()
                result = self._detector.detect(
                    frame,
                    frame_number=self._frame_count,
                    source=source_name,
                )
                result = self._apply_manual_ring_policy(result)
            if self._tracker is not None:
                smoothed = self._tracker.update(result)
            else:
                smoothed = result
            if smoothed.has_both and self._corneal_calc is not None:
                smoothed.corneal_center = self._corneal_calc.calculate(
                    smoothed.pupil,
                    smoothed.limbus,
                    result.calibration,
                )
            smoothed.calibration = result.calibration
            smoothed.ring_status = getattr(result, "ring_status", "unknown")
            smoothed.ring_center = getattr(result, "ring_center", None)
            smoothed.ring_radius = getattr(result, "ring_radius", None)
            smoothed.ring_contour = getattr(result, "ring_contour", None)
            smoothed.ring_dot_count = getattr(result, "ring_dot_count", 0)
            smoothed.corneal_reference_source = getattr(
                result,
                "corneal_reference_source",
                "limbus",
            )
            if (
                smoothed.ring_status == "ring_present"
                and smoothed.ring_center is not None
                and getattr(smoothed, "pupil", None) is not None
                and getattr(smoothed.pupil, "ellipse", None) is not None
            ):
                px = smoothed.pupil.ellipse.center_x
                py = smoothed.pupil.ellipse.center_y
                points = [(px, py, "pupil")]
                weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]
                if getattr(smoothed, "limbus", None) is not None and getattr(smoothed.limbus, "ellipse", None) is not None:
                    points.append(
                        (
                            smoothed.limbus.ellipse.center_x,
                            smoothed.limbus.ellipse.center_y,
                            "limbus",
                        )
                    )
                    weights.append(max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3))
                points.append((smoothed.ring_center[0], smoothed.ring_center[1], "ring"))
                weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                total_w = sum(weights)
                rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                smoothed.corneal_center.center_px = (rcx, rcy)
                smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                smoothed.corneal_center.offset_magnitude_px = math.hypot(
                    px - rcx,
                    py - rcy,
                )
                smoothed.corneal_center.offset_angle_deg = math.degrees(
                    math.atan2(py - rcy, px - rcx)
                )
                smoothed.corneal_center.valid = True
                if result.calibration.calibrated:
                    smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm((rcx, rcy))
                    dx_mm = (px - rcx) * result.calibration.mm_per_px
                    dy_mm = (py - rcy) * result.calibration.mm_per_px
                    smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
                    smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)
            self._current_result = smoothed
            self._results_history.append(smoothed.to_dict())

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, smoothed)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            now = time.monotonic()
            display_interval = self._get_display_interval()
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_classic_frame, smoothed)
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_classic_frame(self, result: Any) -> None:
        self._update_measurements(result)
        self._fps_var.set("---")
        self._refresh_display()

    def _start_video_optimized(self, engine: Any, src_name: str) -> None:
        model_path = self._find_model_path() or "models/best_model.pth"
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized video startup failed, falling back to classic: %s", exc)
            self._opt_processor = None
            self._using_optimized_camera = False
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized video startup failed - using classic pipeline")
            self._start_video_classic(src_name)
            return
        self._using_optimized_camera = True
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set(f"Playing (optimised): {src_name}")
        self._pipeline_var.set(f"Optimised [{preset_label}]")
        self._video_thread = threading.Thread(
            target=self._video_loop_optimized,
            args=(src_name,),
            daemon=True,
            name="VideoLoopOptimised",
        )
        self._video_thread.start()

    def _video_loop_optimized(self, source_name: str) -> None:
        stride = max(1, self._stride_var.get())
        raw_frame_idx = 0
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        # Use decode-ahead threading for video files (not camera)
        if not self._camera_mode and self._video_cap is not None:
            import queue as _queue

            frame_queue = _queue.Queue(
                maxsize=max(6, self._runtime_profile.recommended_capture_buffer * 3)
            )
            from pupil_tracking.video.optimized_processor import _FrameReader

            reader = _FrameReader(self._video_cap, frame_queue, stride=stride)
            reader.start()

            try:
                while self._video_running:
                    if self._video_paused:
                        time.sleep(0.05)
                        continue
                    try:
                        item = frame_queue.get(timeout=5.0)
                    except _queue.Empty:
                        break
                    if item is None:
                        # Video complete
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
                    pending_end = False
                    if (
                        self._opt_processor is not None
                        and self._opt_processor.should_shed_input_frames()
                        and frame_queue.qsize() > 3
                    ):
                        try:
                            queued_item = frame_queue.get_nowait()
                        except _queue.Empty:
                            queued_item = None
                        if queued_item is None:
                            pending_end = True
                        else:
                            item = queued_item
                    raw_frame_idx, frame = item
                    if self._opt_processor is not None:
                        self._opt_processor.note_source_frame(raw_frame_idx)
                    self._frame_count += 1
                    self._current_image = frame
                    try:
                        frame_result = self._opt_processor.process_frame(
                            frame, self._frame_count
                        )
                    except Exception as exc:
                        self.logger.error("Optimised video frame error: %s", exc)
                        continue
                    try:
                        fr_ns = self._dict_to_frame_ns(frame_result)
                        adapted = self._adapt_frame_result(fr_ns, frame.shape)
                    except Exception as exc:
                        self.logger.error("Optimised adapt error: %s", exc)
                        continue
                    self._current_result = adapted
                    try:
                        self._results_history.append(adapted.to_dict())
                    except Exception as exc:
                        self.logger.error("Optimised to_dict error: %s", exc)

                    # ══════════════════════════════════════════════════════════
                    # RECORDING — Write frame to recorder at full resolution
                    # ══════════════════════════════════════════════════════════
                    if self._recorder.is_recording:
                        annotated = self._prepare_recording_frame(frame, adapted)
                        self._write_frame_to_recorder(annotated)
                    # ══════════════════════════════════════════════════════════

                    fps_counter += 1
                    now = time.monotonic()
                    display_interval = self._get_display_interval()
                    elapsed_fps = now - fps_timer
                    if elapsed_fps >= 1.0:
                        current_fps = fps_counter / elapsed_fps
                        fps_counter = 0
                        fps_timer = now
                    if (now - self._last_display_update) >= display_interval:
                        self._last_display_update = now
                        _fps = current_fps
                        self.root.after(
                            0,
                            self._on_optimized_video_frame,
                            adapted,
                            _fps,
                        )
                    self.root.after(0, self._update_progress, raw_frame_idx, True)
                    if pending_end:
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
            finally:
                reader.stop()
                reader.join(timeout=3.0)

            # If loop ended naturally (not via stop button), signal completion
            if not self._video_running:
                return
            self._video_running = False
            self.root.after(
                0,
                lambda: self._status_var.set(
                    f"Video complete (optimised): {len(self._results_history)} frames"
                ),
            )
            self.root.after(0, self._on_video_complete)
            return

        # Fallback: synchronous read (camera via _start_video path)
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                self._video_running = False
                self.root.after(
                    0,
                    lambda: self._status_var.set(
                        f"Video complete (optimised): "
                        f"{len(self._results_history)} frames"
                    ),
                )
                self.root.after(0, self._on_video_complete)
                break
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(raw_frame_idx)
            self._frame_count += 1
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(
                    frame, self._frame_count
                )
            except Exception as exc:
                self.logger.error("Optimised video frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                _fps = current_fps
                self.root.after(
                    0,
                    self._on_optimized_video_frame,
                    adapted,
                    _fps,
                )
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_optimized_video_frame(self, adapted: Any, fps: float) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Video ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set(f"{fps:.1f}")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._refresh_display()

    def _update_progress(self, raw_frame_idx: int, processed: bool) -> None:
        total = self._video_total_frames
        if total <= 0:
            self._progress_label_var.set(f"Frame {raw_frame_idx}")
            return
        self._progress_bar["value"] = raw_frame_idx
        pct = raw_frame_idx / total * 100.0
        self._progress_label_var.set(
            f"Frame {raw_frame_idx}/{total}  ({pct:.1f}%)  —  "
            f"{len(self._results_history)} processed"
        )
        elapsed = time.monotonic() - self._video_start_time
        if raw_frame_idx > 0 and elapsed > 0.5:
            remaining_frames = total - raw_frame_idx
            rate = raw_frame_idx / elapsed
            if rate > 0:
                eta_sec = remaining_frames / rate
                if eta_sec < 60:
                    self._eta_label_var.set(f"ETA: {eta_sec:.0f}s")
                elif eta_sec < 3600:
                    m, s = divmod(int(eta_sec), 60)
                    self._eta_label_var.set(f"ETA: {m}m {s}s")
                else:
                    h, rem = divmod(int(eta_sec), 3600)
                    m, s = divmod(rem, 60)
                    self._eta_label_var.set(f"ETA: {h}h {m}m")
            else:
                self._eta_label_var.set("")
        else:
            self._eta_label_var.set("")

    def _on_video_complete(self) -> None:
        total = self._video_total_frames
        if total > 0:
            self._progress_bar["value"] = total
        elapsed = time.monotonic() - self._video_start_time
        n = len(self._results_history)
        avg_fps = n / elapsed if elapsed > 0 else 0
        self._progress_label_var.set(
            f"Done — {n} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)"
        )
        self._eta_label_var.set("✓ Complete")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")

    def _start_camera(self) -> None:
        self._stop_video()
        self._camera_mode = True
        engine = self._get_fast_engine() if self._use_optimized_var.get() else None
        if engine is not None:
            self._start_camera_optimized(engine)
        else:
            self._start_video(0)

    def _start_camera_optimized(self, engine: Any) -> None:
        try:
            self._async_capture = AsyncCapture(
                0,
                buffer_size=self._runtime_profile.recommended_capture_buffer,
            )
            self._async_capture.start()
        except RuntimeError as exc:
            messagebox.showerror("Camera Error", str(exc))
            return
        self._active_source = 0
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=self._find_model_path() or "models/best_model.pth",
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized camera startup failed, falling back to classic: %s", exc)
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized camera startup failed - using classic pipeline")
            self._opt_processor = None
            if self._async_capture is not None:
                try:
                    self._async_capture.stop()
                except Exception:
                    pass
                self._async_capture = None
            self._using_optimized_camera = False
            self._start_video(0)
            return
        self._using_optimized_camera = True
        self._video_running = True
        self._video_paused = False
        self._frame_count = 0
        self._results_history.clear()
        self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set("Camera (optimised) — starting…")
        self._pipeline_var.set(f"Optimised [{preset_label}]")
        self._progress_bar.config(mode="indeterminate")
        self._progress_label_var.set("Live camera")
        self._eta_label_var.set("")
        self._video_thread = threading.Thread(
            target=self._camera_loop_optimized,
            daemon=True,
            name="OptCameraLoop",
        )
        self._video_thread.start()

    def _camera_loop_optimized(self) -> None:
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        while self._video_running and self._async_capture is not None:
            if self._video_paused:
                time.sleep(0.05)
                continue

            # Check for capture errors
            cap_err = self._async_capture.get_error()
            if cap_err is not None:
                self.logger.error("Camera capture error: %s", cap_err)
                self.root.after(
                    0,
                    lambda e=str(cap_err): self._status_var.set(f"Camera error: {e}"),
                )
                break

            data = self._async_capture.read(timeout=0.05)
            if data is None:
                continue
            fnum, frame, _ts = data
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(fnum)

            # Skip stale frames aggressively to keep surgical UI responsive.
            frame_age = time.time() - _ts
            stale_threshold = (
                self._opt_processor.get_stale_frame_threshold_s()
                if self._opt_processor is not None
                else 0.12
            )
            if frame_age > stale_threshold:
                if self._opt_processor is not None:
                    self._opt_processor.note_stale_frame()
                continue

            self._frame_count = fnum
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(frame, fnum)
            except Exception as exc:
                self.logger.error("Optimised frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised camera adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised camera to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write frame to recorder at full resolution
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(annotated)
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now

            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_optimized_frame, adapted, current_fps)

    def _on_optimized_frame(self, adapted: SimpleNamespace, fps: float = 0.0) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Camera ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set("---")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._progress_label_var.set(f"Frame {self._frame_count}")
        self._refresh_display()

    def _stop_video(self) -> None:
        self._video_running = False
        self._video_paused = False
        self._camera_mode = False
        if not self._restart_in_progress:
            self._active_source = None
        self._roi_edit_active = False
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_original_before_edit = None
        self._roi_preview = None
        self._canvas.configure(cursor="crosshair")
        if hasattr(self, "_roi_btn"):
            self._roi_btn.config(text="Set ROI")
        if self._video_thread is not None:
            try:
                if threading.current_thread() is not self._video_thread:
                    self._video_thread.join(timeout=2.5)
            except Exception:
                pass
            self._video_thread = None
        if self._async_capture is not None:
            try:
                self._async_capture.stop()
            except Exception:
                pass
            self._async_capture = None
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        if self._opt_processor is not None:
            try:
                self._opt_processor.reset()
            except Exception:
                pass
            self._opt_processor = None
        self._last_opt_stats = {}
        self._using_optimized_camera = False
        self._pipeline_var.set("---")
        self._fps_var.set("---")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self._progress_bar.config(mode="determinate")
        self._progress_bar["value"] = 0
        # ══════════════════════════════════════════════════════════
        # RECORDING — Auto-stop recording when video stops
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()

    def _run_iris_detection(
        self, image: np.ndarray, result: EyeDetectionResult
    ) -> EyeDetectionResult:
        """Run iris feature detection if pupil/limbus are available.

        This is ADDITIVE — it does not modify the existing detection result.
        Iris data is attached as dynamic attributes on the result.
        """
        try:
            if not result.has_both:
                result.iris_status = None
                result.iris_detection = None
                return result

            pupil = result.pupil.ellipse
            limbus = result.limbus.ellipse
            if pupil is None or limbus is None:
                result.iris_status = None
                result.iris_detection = None
                return result

            iris_result = detect_iris_features(image, pupil, limbus)
            result.iris_detection = iris_result
            result.iris_status = iris_result.status
            return result
        except Exception as exc:
            self.logger.debug("Iris detection skipped: %s", exc)
            result.iris_status = None
            result.iris_detection = None
            return result

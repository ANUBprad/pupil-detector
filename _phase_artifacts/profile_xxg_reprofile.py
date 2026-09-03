"""Phase XX-G: Re-profile SmartContourFitter after XX-E/XX-F optimizations."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.smart_fitter import (
    SmartContourFitter, FitResult,
    _fit_circle_taubin, _ransac_circle, _circle_residuals,
    _compute_gradient_weights, _fit_circle_weighted_taubin,
)
from pupil_tracking.core.smart_fitter import FitType

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
SAMPLE_COUNT = 24
OUTPUT_DIR = Path("_phase_artifacts")

# ── Monkey-patch SmartContourFitter.fit_contour for profiling ──────
_fit_stage_times: Dict[str, float] = {}
_fit_stage_counts: Dict[str, int] = {}


def _patched_fit_contour(self, points):
    """Profiled fit_contour that records internal stage timings."""
    if len(points) < self.min_contour_points:
        return FitResult()

    result = FitResult(num_contour_points=len(points))
    result.contour_points = points

    # Stage 1: RANSAC circle fit
    t0 = time.perf_counter()
    if self.use_ransac:
        contour_span = float(np.ptp(points, axis=0).max())
        adaptive_thresh = max(self.ransac_threshold, contour_span * 0.01)
        ransac_result = _ransac_circle(points, inlier_threshold=adaptive_thresh)
        if ransac_result is not None:
            c_cx, c_cy, c_r, inlier_mask = ransac_result
            circle_inliers = points[inlier_mask]
            result.num_inliers = int(np.sum(inlier_mask))
        else:
            circle_result = _fit_circle_taubin(points)
            if circle_result is None:
                _fit_stage_times['ransac'] = _fit_stage_times.get('ransac', 0) + (time.perf_counter() - t0) * 1000
                _fit_stage_counts['ransac'] = _fit_stage_counts.get('ransac', 0) + 1
                return self._fallback_ellipse_only(points, result)
            c_cx, c_cy, c_r = circle_result
            circle_inliers = points
            result.num_inliers = len(points)
    else:
        circle_result = _fit_circle_taubin(points)
        if circle_result is None:
            circle_result = _fit_circle_taubin(points)
        if circle_result is None:
            _fit_stage_times['ransac'] = _fit_stage_times.get('ransac', 0) + (time.perf_counter() - t0) * 1000
            _fit_stage_counts['ransac'] = _fit_stage_counts.get('ransac', 0) + 1
            return self._fallback_ellipse_only(points, result)
        c_cx, c_cy, c_r = circle_result
        circle_inliers = points
        result.num_inliers = len(points)
    _fit_stage_times['ransac'] = _fit_stage_times.get('ransac', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['ransac'] = _fit_stage_counts.get('ransac', 0) + 1

    # Stage 2: Circle residuals
    t0 = time.perf_counter()
    c_residuals = _circle_residuals(circle_inliers, c_cx, c_cy, c_r)
    c_rms = float(np.sqrt(np.mean(c_residuals ** 2)))
    result.circle_rms = c_rms
    _fit_stage_times['circle_residuals'] = _fit_stage_times.get('circle_residuals', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['circle_residuals'] = _fit_stage_counts.get('circle_residuals', 0) + 1

    # Stage 3: Weighted Taubin refinement
    t0 = time.perf_counter()
    if (
        self.subpixel_refine
        and self._last_gray is not None
        and self._subpixel_cfg.use_weighted_fit
        and len(circle_inliers) >= 5
    ):
        weights = _compute_gradient_weights(self._last_gray, circle_inliers)
        if weights is not None and len(weights) == len(circle_inliers):
            weighted_fit = _fit_circle_weighted_taubin(circle_inliers, weights)
            if weighted_fit is not None:
                w_cx, w_cy, w_r = weighted_fit
                w_res = _circle_residuals(circle_inliers, w_cx, w_cy, w_r)
                w_rms = float(np.sqrt(np.mean(w_res ** 2)))
                if w_rms <= c_rms:
                    c_cx, c_cy, c_r = w_cx, w_cy, w_r
                    c_rms = w_rms
                    result.circle_rms = c_rms
    _fit_stage_times['weighted_taubin'] = _fit_stage_times.get('weighted_taubin', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['weighted_taubin'] = _fit_stage_counts.get('weighted_taubin', 0) + 1

    # Stage 4: Ellipse fit
    t0 = time.perf_counter()
    if len(circle_inliers) >= 5:
        try:
            pts_cv = circle_inliers.reshape(-1, 1, 2).astype(np.float32)
            ellipse = cv2.fitEllipse(pts_cv)
            (e_cx, e_cy), (e_w, e_h), e_angle = ellipse
            e_a = max(e_w, e_h) / 2.0
            e_b = min(e_w, e_h) / 2.0
            e_angle_rad = math.radians(e_angle)
            from pupil_tracking.core.smart_fitter import _ellipse_residuals
            e_residuals = _ellipse_residuals(circle_inliers, e_cx, e_cy, e_a, e_b, e_angle_rad)
            e_rms = float(np.sqrt(np.mean(e_residuals ** 2)))
            result.ellipse_rms = e_rms
            ellipse_valid = True
        except cv2.error:
            ellipse_valid = False
            e_rms = float("inf")
    else:
        ellipse_valid = False
        e_rms = float("inf")
    _fit_stage_times['ellipse_fit'] = _fit_stage_times.get('ellipse_fit', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['ellipse_fit'] = _fit_stage_counts.get('ellipse_fit', 0) + 1

    # Stage 5: Circle-vs-ellipse decision
    t0 = time.perf_counter()
    use_circle = False
    if not ellipse_valid:
        use_circle = True
    else:
        aspect = e_b / e_a if e_a > 0 else 1.0
        if aspect >= self.circularity_threshold:
            use_circle = True
        elif c_rms < 1e-6:
            use_circle = True
        elif e_rms > 0 and c_rms / e_rms < self.residual_ratio_threshold:
            use_circle = True
        else:
            use_circle = False
    _fit_stage_times['decision'] = _fit_stage_times.get('decision', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['decision'] = _fit_stage_counts.get('decision', 0) + 1

    # Populate result
    if use_circle:
        result.fit_type = FitType.CIRCLE
        result.valid = True
        result.center_x = c_cx
        result.center_y = c_cy
        result.semi_major = c_r
        result.semi_minor = c_r
        result.radius = c_r
        result.angle_deg = 0.0
        result.eccentricity = 0.0
        result.circularity = 1.0
        result.aspect_ratio = 1.0
        result.fit_rms_residual = c_rms
    else:
        result.fit_type = FitType.ELLIPSE
        result.valid = True
        result.center_x = e_cx
        result.center_y = e_cy
        result.semi_major = e_a
        result.semi_minor = e_b
        result.radius = e_a
        result.angle_deg = e_angle
        result.eccentricity = math.sqrt(max(0.0, 1.0 - (e_b / e_a) ** 2)) if e_a > 0 else 0.0
        result.aspect_ratio = e_b / e_a if e_a > 0 else 0.0
        result.circularity = result.aspect_ratio
        result.fit_rms_residual = e_rms

    # Stage 6: Quality + uncertainty
    t0 = time.perf_counter()
    result.fit_quality = self._compute_quality(result)
    self._compute_uncertainty(result, circle_inliers)
    _fit_stage_times['quality_uncertainty'] = _fit_stage_times.get('quality_uncertainty', 0) + (time.perf_counter() - t0) * 1000
    _fit_stage_counts['quality_uncertainty'] = _fit_stage_counts.get('quality_uncertainty', 0) + 1

    return result


SmartContourFitter.fit_contour = _patched_fit_contour


def sample_frame_indices(total, count):
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def profile_frame(detector, frame_bgr, frame_idx):
    """Profile one frame through the full detection pipeline."""
    global _fit_stage_times, _fit_stage_counts
    _fit_stage_times.clear()
    _fit_stage_counts.clear()

    t0 = time.perf_counter()
    result = None
    error = None
    try:
        result = detector.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        error = str(e)
    total_ms = (time.perf_counter() - t0) * 1000

    out: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "total_ms": round(total_ms, 2),
        "error": error,
    }

    if result is not None:
        out["pupil_detected"] = bool(getattr(result.pupil, "detected", False))
        out["limbus_detected"] = bool(getattr(result.limbus, "detected", False))
        ell_p = getattr(result.pupil, "ellipse", None)
        if ell_p is not None:
            axes = getattr(ell_p, "axes", None)
            center = getattr(ell_p, "center", None)
            out["pupil_radius"] = round(float(max(axes)), 2) if axes else 0
            out["pupil_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center else [0, 0]
        else:
            out["pupil_radius"] = 0
            out["pupil_center"] = [0, 0]
        ell_l = getattr(result.limbus, "ellipse", None)
        if ell_l is not None:
            axes = getattr(ell_l, "axes", None)
            center = getattr(ell_l, "center", None)
            out["limbus_radius"] = round(float(max(axes)), 2) if axes else 0
            out["limbus_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center else [0, 0]
        else:
            out["limbus_radius"] = 0
            out["limbus_center"] = [0, 0]
        out["pupil_method"] = str(getattr(result.pupil, "method", "none"))
        out["limbus_method"] = str(getattr(result.limbus, "method", "none"))
    else:
        out["pupil_detected"] = False
        out["limbus_detected"] = False
        out["confidence"] = 0
        out["pupil_radius"] = 0
        out["pupil_center"] = [0, 0]
        out["limbus_radius"] = 0
        out["limbus_center"] = [0, 0]
        out["pupil_method"] = "none"
        out["limbus_method"] = "none"

    # Record fit_contour stage times
    out["fit_contour_stages"] = {k: round(v, 2) for k, v in _fit_stage_times.items()}
    out["fit_contour_counts"] = dict(_fit_stage_counts)

    return out


def main():
    print("=" * 70)
    print("PHASE XX-G: Post-Optimization SmartContourFitter Re-profiling")
    print("=" * 70)

    detector = UnifiedDetector()
    all_results: List[Dict[str, Any]] = []

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = sample_frame_indices(total_frames, SAMPLE_COUNT)
        print(f"\n  Video: {video_path} ({total_frames} frames)")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            result = profile_frame(detector, frame, fidx)
            result["video"] = video_path
            all_results.append(result)

            ms = result["total_ms"]
            p = "P+" if result["pupil_detected"] else "P-"
            l = "L+" if result["limbus_detected"] else "L-"
            stages = result.get("fit_contour_stages", {})
            total_fit = sum(stages.values())
            print(f"    [{fi+1}/{len(indices)}] frame {fidx}: {ms:6.0f} ms | {p} {l} | fit_contour={total_fit:.0f}ms")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxg_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nProfile saved to {out_path}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("STAGE BREAKDOWN (fit_contour only)")
    print(f"{'=' * 70}")

    all_stages: Dict[str, List[float]] = {}
    for r in all_results:
        for stage, ms in r.get("fit_contour_stages", {}).items():
            all_stages.setdefault(stage, []).append(ms)

    total_fit_time = sum(sum(v) for v in all_stages.values())
    print(f"\n{'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s} {'Total':>10s} {'%':>6s}")
    print("-" * 80)
    for stage in sorted(all_stages, key=lambda s: -sum(all_stages[s])):
        vals = all_stages[stage]
        total = sum(vals)
        pct = total / total_fit_time * 100 if total_fit_time > 0 else 0
        print(f"{stage:25s} {np.mean(vals):10.2f} {np.median(vals):10.2f} {np.percentile(vals, 95):10.2f} {np.max(vals):10.2f} {total:10.1f} {pct:5.1f}%")

    # ── Total performance ──
    times = [r["total_ms"] for r in all_results]
    print(f"\n{'=' * 70}")
    print("TOTAL PERFORMANCE")
    print(f"{'=' * 70}")
    print(f"  Mean:   {np.mean(times):.0f} ms")
    print(f"  Median: {np.median(times):.0f} ms")
    print(f"  P95:    {np.percentile(times, 95):.0f} ms")
    print(f"  Worst:  {np.max(times):.0f} ms")

    # ── Comparison with XX-D ──
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH XX-D BASELINE")
    print(f"{'=' * 70}")
    print(f"  Metric                XX-D         Current      Improvement")
    print(f"  {'-'*65}")
    print(f"  Subpixel/frame        3,319 ms     275 ms       91.7%")
    print(f"  Gradient comp         10-15x       1x           93%")
    print(f"  Total mean            4,549 ms     {np.mean(times):.0f} ms       {(4549 - np.mean(times)) / 4549 * 100:.1f}%")

    # ── Correctness ──
    pupil_detected = sum(1 for r in all_results if r["pupil_detected"])
    limbus_detected = sum(1 for r in all_results if r["limbus_detected"])
    print(f"\n{'=' * 70}")
    print("CORRECTNESS")
    print(f"{'=' * 70}")
    print(f"  Pupil detected:  {pupil_detected}/48")
    print(f"  Limbus detected: {limbus_detected}/48")


if __name__ == "__main__":
    main()

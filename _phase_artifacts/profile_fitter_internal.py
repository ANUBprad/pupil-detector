"""Phase XX-D: SmartContourFitter internal profiling on 48 ELITA frames.

Instruments SmartContourFitter.fit() to measure:
- contour extraction time
- subpixel refinement time
- circle fitting (RANSAC + Taubin + Hyper) time
- ellipse fitting time
- circle-vs-ellipse decision time
- quality/uncertainty time
- candidate scoring (in caller) time
- per-iteration and per-frame breakdowns
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ─── Monkey-patch SmartContourFitter for profiling ───────────────
from pupil_tracking.core import smart_fitter as sf_module
from pupil_tracking.core.smart_fitter import (
    SmartContourFitter,
    FitResult,
    _fit_circle_taubin,
    _fit_circle_hyper,
    _fit_circle_kasa,
    _ransac_circle,
    _circle_residuals,
    _ellipse_residuals,
    _compute_gradient_weights,
    _fit_circle_weighted_taubin,
    _compute_multiscale_gradient,
    _refine_contour_subpixel,
)

# Global accumulator for per-fit stage timings
_fit_stages: Dict[str, float] = {}
_fit_calls = 0


def _patched_fit(self, binary_mask, gray_image=None, pupil_hint=None):
    """Patched fit() that records internal stage timings."""
    global _fit_calls
    _fit_calls += 1

    t_total = time.perf_counter()

    # Stage 1: mask preparation
    t0 = time.perf_counter()
    mask = binary_mask.copy()
    if mask.max() == 1:
        mask = mask * 255
    _fit_stages['mask_prep'] = _fit_stages.get('mask_prep', 0) + (time.perf_counter() - t0)

    # Stage 2: contour extraction
    t0 = time.perf_counter()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _fit_stages['contour_extract'] = _fit_stages.get('contour_extract', 0) + (time.perf_counter() - t0)

    if not contours:
        _fit_stages['total'] = _fit_stages.get('total', 0) + (time.perf_counter() - t_total)
        return FitResult()

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < self.min_area:
        _fit_stages['total'] = _fit_stages.get('total', 0) + (time.perf_counter() - t_total)
        return FitResult()

    pts = largest.reshape(-1, 2).astype(np.float64)
    if len(pts) < self.min_contour_points:
        _fit_stages['total'] = _fit_stages.get('total', 0) + (time.perf_counter() - t_total)
        return FitResult()

    # Stage 3: pupil_hint filtering
    t0 = time.perf_counter()
    if pupil_hint is not None and pupil_hint.valid:
        dx = pts[:, 0] - pupil_hint.center_x
        dy = pts[:, 1] - pupil_hint.center_y
        distances = np.hypot(dx, dy)
        median_dist = np.median(distances)
        upper_bound = median_dist * 1.15
        lower_bound = median_dist * 0.85
        mask_pts = (distances >= lower_bound) & (distances <= upper_bound)
        if np.sum(mask_pts) >= max(self.min_contour_points, int(len(pts) * 0.25)):
            pts = pts[mask_pts]
    _fit_stages['hint_filter'] = _fit_stages.get('hint_filter', 0) + (time.perf_counter() - t0)

    # Stage 4: subpixel refinement
    t0 = time.perf_counter()
    if self.subpixel_refine and gray_image is not None:
        sp = self._subpixel_cfg
        pts = _refine_contour_subpixel(
            gray_image, pts,
            use_multiscale=sp.use_multiscale_gradient,
            interpolation_step=sp.interpolation_step,
            use_parabolic=sp.use_parabolic_peak,
        )
        self._last_gray = gray_image
    else:
        self._last_gray = None
    _fit_stages['subpixel'] = _fit_stages.get('subpixel', 0) + (time.perf_counter() - t0)

    # Now call fit_contour with timing
    result = _patched_fit_contour(self, pts)

    _fit_stages['total'] = _fit_stages.get('total', 0) + (time.perf_counter() - t_total)
    return result


def _patched_fit_contour(self, points):
    """Patched fit_contour() that records internal stage timings."""
    if len(points) < self.min_contour_points:
        return FitResult()

    result = FitResult(num_contour_points=len(points))
    result.contour_points = points

    # Stage 5: RANSAC circle fit
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
                _fit_stages['circle_fit'] = _fit_stages.get('circle_fit', 0) + (time.perf_counter() - t0)
                _fit_stages['total'] = _fit_stages.get('total', 0) + (time.perf_counter() - t0)
                return self._fallback_ellipse_only(points, result)
            c_cx, c_cy, c_r = circle_result
            circle_inliers = points
            result.num_inliers = len(points)
    else:
        circle_result = _fit_circle_hyper(points)
        if circle_result is None:
            circle_result = _fit_circle_taubin(points)
        if circle_result is None:
            _fit_stages['circle_fit'] = _fit_stages.get('circle_fit', 0) + (time.perf_counter() - t0)
            return self._fallback_ellipse_only(points, result)
        c_cx, c_cy, c_r = circle_result
        circle_inliers = points
        result.num_inliers = len(points)
    _fit_stages['circle_fit'] = _fit_stages.get('circle_fit', 0) + (time.perf_counter() - t0)

    # Circle residuals
    t0 = time.perf_counter()
    c_residuals = _circle_residuals(circle_inliers, c_cx, c_cy, c_r)
    c_rms = float(np.sqrt(np.mean(c_residuals ** 2)))
    result.circle_rms = c_rms
    _fit_stages['circle_residuals'] = _fit_stages.get('circle_residuals', 0) + (time.perf_counter() - t0)

    # Stage 6: weighted Taubin refinement
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
    _fit_stages['weighted_taubin'] = _fit_stages.get('weighted_taubin', 0) + (time.perf_counter() - t0)

    # Stage 7: ellipse fit
    t0 = time.perf_counter()
    if len(circle_inliers) >= 5:
        try:
            pts_cv = circle_inliers.reshape(-1, 1, 2).astype(np.float32)
            ellipse = cv2.fitEllipse(pts_cv)
            (e_cx, e_cy), (e_w, e_h), e_angle = ellipse
            e_a = max(e_w, e_h) / 2.0
            e_b = min(e_w, e_h) / 2.0
            e_angle_rad = math.radians(e_angle)
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
    _fit_stages['ellipse_fit'] = _fit_stages.get('ellipse_fit', 0) + (time.perf_counter() - t0)

    # Stage 8: circle-vs-ellipse decision
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
    _fit_stages['decision'] = _fit_stages.get('decision', 0) + (time.perf_counter() - t0)

    # Stage 9: populate result
    if use_circle:
        result.fit_type = sf_module.FitType.CIRCLE
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
        result.fit_type = sf_module.FitType.ELLIPSE
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

    # Stage 10: quality + uncertainty
    t0 = time.perf_counter()
    result.fit_quality = self._compute_quality(result)
    self._compute_uncertainty(result, circle_inliers)
    _fit_stages['quality_uncertainty'] = _fit_stages.get('quality_uncertainty', 0) + (time.perf_counter() - t0)

    return result


# Apply patches
SmartContourFitter.fit = _patched_fit
SmartContourFitter.fit_contour = _patched_fit_contour

# ─── Classical pupil fallback profiling ───────────────────────────

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.deterministic_ring_detector import RingStatus
from pupil_tracking.utils.types import DetectionMethod, PupilDetection

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
SAMPLE_COUNT = 24
OUTPUT_DIR = Path("_phase_artifacts")


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
    """Profile one frame through the classical pupil fallback."""
    global _fit_stages, _fit_calls
    _fit_stages.clear()
    _fit_calls = 0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_diag = math.sqrt(h * h + w * w)
    min_radius = max(8, int(img_diag * 0.015))
    max_radius = int(img_diag * 0.25)
    min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    threshold_iterations: List[Dict[str, Any]] = []
    best_fit = None
    best_score = 0.0
    best_contour = None

    for pct in [3, 5, 8, 12, 18, 25]:
        t_iter = time.perf_counter()

        t_thresh = time.perf_counter()
        thresh_val = np.percentile(blurred, pct)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        t_thresh_ms = (time.perf_counter() - t_thresh) * 1000

        t_contours = time.perf_counter()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        t_contours_ms = (time.perf_counter() - t_contours) * 1000

        iter_fitter_calls = 0
        iter_fitter_ms = 0.0
        iter_valid = 0
        iter_candidates: List[Dict[str, Any]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or len(cnt) < 15:
                continue

            t_fit_start = time.perf_counter()
            cnt_mask = np.zeros_like(gray)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
            fit = detector._fitter.fit(cnt_mask, gray)
            t_fit_ms = (time.perf_counter() - t_fit_start) * 1000

            iter_fitter_calls += 1
            iter_fitter_ms += t_fit_ms

            if fit is None or not fit.valid:
                continue
            if fit.radius < min_radius or fit.radius > max_radius:
                continue

            iter_valid += 1

            # Scoring (same as detector.py)
            centrality = max(0.0, 1.0 - (
                abs(fit.center_x - w/2) / (w/2) * 0.5
                + abs(fit.center_y - h/2) / (h/2) * 0.5
            ))
            circ = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0
            mask_tmp = np.zeros_like(gray)
            cv2.drawContours(mask_tmp, [cnt], -1, 255, -1)
            darkness = 1.0 - (cv2.mean(gray, mask=mask_tmp)[0] / 255.0)
            fit_quality = fit.fit_quality if fit.fit_quality is not None else 0.5

            score = (
                0.25 * centrality
                + 0.25 * min(1.0, circ / 0.7)
                + 0.25 * fit_quality
                + 0.25 * darkness
            )

            if score > best_score:
                best_score = score
                best_fit = fit
                best_contour = cnt

            iter_candidates.append({
                "radius": round(float(fit.radius), 2),
                "center": [round(float(fit.center_x), 1), round(float(fit.center_y), 1)],
                "score": round(score, 4),
                "fitter_ms": round(t_fit_ms, 2),
                "contour_points": len(cnt),
            })

        iter_time = (time.perf_counter() - t_iter) * 1000

        threshold_iterations.append({
            "percentile": pct,
            "time_ms": round(iter_time, 2),
            "threshold_ms": round(t_thresh_ms, 2),
            "contours_ms": round(t_contours_ms, 2),
            "fitter_calls": iter_fitter_calls,
            "fitter_ms": round(iter_fitter_ms, 2),
            "candidates_valid": iter_valid,
            "candidates": iter_candidates[:5],  # top 5 only
        })

    total_time = sum(i["time_ms"] for i in threshold_iterations)

    return {
        "frame_idx": frame_idx,
        "total_ms": round(total_time, 2),
        "iterations": threshold_iterations,
        "fit_calls_total": _fit_calls,
        "fit_stages": {k: round(v * 1000, 2) for k, v in _fit_stages.items()},
        "best_score": round(best_score, 4),
        "best_detected": best_fit is not None and best_score > 0.20,
        "pupil_center": [round(float(best_fit.center_x), 2), round(float(best_fit.center_y), 2)] if best_fit and best_score > 0.20 else [0, 0],
        "pupil_radius": round(float(best_fit.radius), 2) if best_fit and best_score > 0.20 else 0,
    }


def main():
    print("=" * 70)
    print("PHASE XX-D: SmartContourFitter Internal Profiling")
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
        print(f"\n  Video: {video_path} ({total_frames} frames, sampling {len(indices)})")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            print(f"    [{fi+1}/{len(indices)}] frame {fidx}...", end=" ", flush=True)
            result = profile_frame(detector, frame, fidx)
            result["video"] = video_path
            all_results.append(result)

            ms = result["total_ms"]
            calls = result["fit_calls_total"]
            detected = "P+" if result["best_detected"] else "P-"
            stages = result["fit_stages"]
            top_stage = max(stages, key=stages.get) if stages else "none"
            print(f"{ms:7.0f} ms | {detected} | {calls} fitter calls | top: {top_stage}={stages.get(top_stage, 0):.0f}ms")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxd_fitter_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nProfile results saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_stages: Dict[str, List[float]] = {}
    for r in all_results:
        for stage, ms in r.get("fit_stages", {}).items():
            all_stages.setdefault(stage, []).append(ms)

    print(f"\n{'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s} {'Total':>10s}")
    print("-" * 75)
    for stage in sorted(all_stages, key=lambda s: -sum(all_stages[s])):
        vals = all_stages[stage]
        print(f"{stage:25s} {np.mean(vals):10.1f} {np.median(vals):10.1f} {np.percentile(vals, 95):10.1f} {np.max(vals):10.1f} {sum(vals):10.1f}")

    # Per-iteration breakdown
    print("\n\nTHRESHOLD ITERATION BREAKDOWN:")
    print("-" * 75)
    iter_stages: Dict[str, List[float]] = {}
    for r in all_results:
        for it in r.get("iterations", []):
            for key in ["threshold_ms", "contours_ms", "fitter_ms"]:
                iter_stages.setdefault(key, []).append(it.get(key, 0))

    print(f"{'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s}")
    print("-" * 75)
    for key in ["threshold_ms", "contours_ms", "fitter_ms"]:
        vals = iter_stages.get(key, [0])
        print(f"{key:25s} {np.mean(vals):10.2f} {np.median(vals):10.2f} {np.percentile(vals, 95):10.2f} {np.max(vals):10.2f}")

    # Candidate counts per iteration
    print("\n\nCANDIDATE COUNTS PER ITERATION:")
    print("-" * 75)
    iter_candidates = {}
    for r in all_results:
        for it in r.get("iterations", []):
            pct = it["percentile"]
            iter_candidates.setdefault(pct, []).append(it.get("candidates_valid", 0))

    for pct in sorted(iter_candidates):
        vals = iter_candidates[pct]
        print(f"  pct={pct:2d}: mean={np.mean(vals):.1f}  median={np.median(vals):.0f}  max={np.max(vals):.0f}")

    # Slow frames
    print("\n\nSLOWEST FRAMES:")
    print("-" * 75)
    sorted_results = sorted(all_results, key=lambda r: -r["total_ms"])
    for r in sorted_results[:10]:
        stages = r.get("fit_stages", {})
        top = max(stages, key=stages.get) if stages else "none"
        print(f"  frame {r['frame_idx']:5d} ({r['video'][:8]}): {r['total_ms']:7.0f} ms | {r['fit_calls_total']} fitter calls | top: {top}={stages.get(top, 0):.0f}ms")


if __name__ == "__main__":
    main()

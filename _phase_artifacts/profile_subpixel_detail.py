"""Phase XX-D: Profile subpixel refinement internals."""
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

from pupil_tracking.core.smart_fitter import (
    _compute_multiscale_gradient,
    _refine_contour_subpixel,
    _fit_circle_taubin,
    _fit_circle_hyper,
    _fit_circle_kasa,
    _ransac_circle,
    _circle_residuals,
)

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
SAMPLE_COUNT = 24


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


def profile_subpixel_detail(gray, contour_pts):
    """Profile each stage of subpixel refinement."""
    h, w = gray.shape[:2]
    results = {}

    # Stage 1: Multi-scale gradient computation
    t0 = time.perf_counter()
    grad_mag, grad_x, grad_y = _compute_multiscale_gradient(gray)
    results['gradient_compute'] = (time.perf_counter() - t0) * 1000

    # Stage 2: Per-point refinement loop
    t0 = time.perf_counter()
    n_steps = int(3 / 0.25)  # search_radius=3, step=0.25
    t_values = np.arange(-n_steps, n_steps + 1) * 0.25

    refined = contour_pts.copy().astype(np.float64)
    for i in range(len(contour_pts)):
        px, py = contour_pts[i]
        ix, iy = int(round(px)), int(round(py))
        if not (1 <= iy < h - 1 and 1 <= ix < w - 1):
            continue
        gx = grad_x[iy, ix]
        gy = grad_y[iy, ix]
        g_len = math.sqrt(gx ** 2 + gy ** 2)
        if g_len < 1e-6:
            continue
        nx, ny = gx / g_len, gy / g_len

        samples = np.zeros(len(t_values))
        for j, t in enumerate(t_values):
            sx = px + nx * t
            sy = py + ny * t
            if not (0 <= sx < w - 1 and 0 <= sy < h - 1):
                continue
            x0, y0 = int(sx), int(sy)
            fx, fy = sx - x0, sy - y0
            samples[j] = (
                grad_mag[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + grad_mag[y0, x0 + 1] * fx * (1.0 - fy)
                + grad_mag[y0 + 1, x0] * (1.0 - fx) * fy
                + grad_mag[y0 + 1, x0 + 1] * fx * fy
            )
        peak_idx = int(np.argmax(samples))
        if 1 <= peak_idx < len(samples) - 1:
            y_m1 = samples[peak_idx - 1]
            y_0 = samples[peak_idx]
            y_p1 = samples[peak_idx + 1]
            denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
            if abs(denom) > 1e-12:
                delta = (y_m1 - y_p1) / denom
                best_t = t_values[peak_idx] + delta * 0.25
            else:
                best_t = t_values[peak_idx]
        else:
            best_t = t_values[peak_idx]
        refined[i, 0] = px + nx * best_t
        refined[i, 1] = py + ny * best_t

    results['per_point_loop'] = (time.perf_counter() - t0) * 1000

    # Stage 3: RANSAC circle fit
    t0 = time.perf_counter()
    contour_span = float(np.ptp(contour_pts, axis=0).max())
    adaptive_thresh = max(2.0, contour_span * 0.01)
    ransac_result = _ransac_circle(contour_pts, inlier_threshold=adaptive_thresh)
    results['ransac_circle'] = (time.perf_counter() - t0) * 1000

    # Stage 4: Taubin fit (on full contour)
    t0 = time.perf_counter()
    _fit_circle_taubin(contour_pts)
    results['taubin_full'] = (time.perf_counter() - t0) * 1000

    # Stage 5: Taubin fit (on RANSAC inliers if available)
    if ransac_result is not None:
        _, _, _, inlier_mask = ransac_result
        inlier_pts = contour_pts[inlier_mask]
        t0 = time.perf_counter()
        _fit_circle_taubin(inlier_pts)
        results['taubin_inliers'] = (time.perf_counter() - t0) * 1000

    results['contour_points'] = len(contour_pts)
    return results


def main():
    print("=" * 70)
    print("PHASE XX-D: Subpixel Refinement Internal Profiling")
    print("=" * 70)

    all_results = []

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            img_diag = math.sqrt(h * h + w * w)
            min_radius = max(8, int(img_diag * 0.015))
            max_radius = int(img_diag * 0.25)
            min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Extract contours from first threshold
            thresh_val = np.percentile(blurred, 3)
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Profile the largest valid contour
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or len(cnt) < 15:
                    continue

                pts = cnt.reshape(-1, 2).astype(np.float64)
                result = profile_subpixel_detail(gray, pts)
                result["frame_idx"] = fidx
                result["video"] = video_path
                all_results.append(result)

                print(f"    frame {fidx:5d}: {result['contour_points']:4d} pts | grad={result['gradient_compute']:.1f}ms loop={result['per_point_loop']:.1f}ms ransac={result['ransac_circle']:.1f}ms taubin={result['taubin_full']:.1f}ms")
                break  # Only profile first valid contour per frame

    # Summary
    print("\n" + "=" * 70)
    print("SUBPIXEL REFINE INTERNAL BREAKDOWN")
    print("=" * 70)

    stages = ['gradient_compute', 'per_point_loop', 'ransac_circle', 'taubin_full']
    print(f"\n{'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s}")
    print("-" * 65)
    for stage in stages:
        vals = [r[stage] for r in all_results if stage in r]
        if vals:
            print(f"{stage:25s} {np.mean(vals):10.2f} {np.median(vals):10.2f} {np.percentile(vals, 95):10.2f} {np.max(vals):10.2f}")

    # Point counts
    pts = [r['contour_points'] for r in all_results]
    print(f"\nContour points: mean={np.mean(pts):.0f}  median={np.median(pts):.0f}  max={np.max(pts)}")

    # Save
    out_path = Path("_phase_artifacts/phase_xxd_subpixel_profile.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

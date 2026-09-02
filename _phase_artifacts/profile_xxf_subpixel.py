"""Phase XX-F: Targeted subpixel refinement benchmark."""
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

from pupil_tracking.core.smart_fitter import SmartContourFitter, _refine_contour_subpixel, _compute_multiscale_gradient

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


def extract_contours(frame_bgr):
    """Extract contours from frame using classical pupil fallback."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_diag = math.sqrt(h * h + w * w)
    min_radius = max(8, int(img_diag * 0.015))
    min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    all_contours = []

    for pct in [3, 5, 8, 12, 18, 25]:
        thresh_val = np.percentile(blurred, pct)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or len(cnt) < 15:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            all_contours.append(pts)

    return gray, all_contours


def main():
    print("=" * 70)
    print("PHASE XX-F: Targeted Subpixel Refinement Benchmark")
    print("=" * 70)

    total_subpixel_before = 0.0
    total_subpixel_after = 0.0
    total_contours = 0
    total_points = 0

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = sample_frame_indices(total_frames, SAMPLE_COUNT)
        print(f"\n  Video: {video_path}")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            gray, contours = extract_contours(frame)
            if not contours:
                continue

            # Compute gradients once (cached)
            grad_mag, grad_x, grad_y = _compute_multiscale_gradient(gray)

            # Time subpixel refinement for all contours
            t0 = time.perf_counter()
            for pts in contours:
                _refine_contour_subpixel(
                    gray, pts,
                    cached_grad_mag=grad_mag,
                    cached_grad_x=grad_x,
                    cached_grad_y=grad_y,
                )
            subpixel_ms = (time.perf_counter() - t0) * 1000

            n_pts = sum(len(pts) for pts in contours)
            total_subpixel_before += subpixel_ms  # This IS the vectorized version
            total_contours += len(contours)
            total_points += n_pts

            print(f"    frame {fidx:5d}: {len(contours):3d} contours, {n_pts:5d} pts, subpixel={subpixel_ms:6.1f}ms")

    print(f"\n{'=' * 70}")
    print(f"TOTALS:")
    print(f"  Contours: {total_contours}")
    print(f"  Points: {total_points}")
    print(f"  Subpixel total: {total_subpixel_before:.0f} ms")
    print(f"  Mean per frame: {total_subpixel_before / 48:.0f} ms")
    print(f"  Mean per contour: {total_subpixel_before / total_contours:.1f} ms")
    print(f"  Mean per point: {total_subpixel_before / total_points * 1000:.2f} us")


if __name__ == "__main__":
    main()

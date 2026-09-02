"""Phase XX-D: Profile bootstrap uncertainty and weighted Taubin."""
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
    _fit_circle_taubin,
    _compute_gradient_weights,
    _fit_circle_weighted_taubin,
    _ransac_circle,
    _circle_residuals,
)

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    print("=" * 70)
    print("PHASE XX-D: Bootstrap + Weighted Taubin Profiling")
    print("=" * 70)

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = [int(i * total_frames / 6) for i in range(6)]
        print(f"\n  Video: {video_path}")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            img_diag = math.sqrt(h * h + w * w)
            min_radius = max(8, int(img_diag * 0.015))
            min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            thresh_val = np.percentile(blurred, 3)
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

                # Bootstrap
                t0 = time.perf_counter()
                rng = np.random.RandomState(42)
                n = len(pts)
                cx_list, cy_list, r_list = [], [], []
                for _ in range(50):
                    idx = rng.choice(n, n, replace=True)
                    sample = pts[idx]
                    fit = _fit_circle_taubin(sample)
                    if fit is not None:
                        cx, cy, r = fit
                        if r > 0 and math.isfinite(cx) and math.isfinite(cy):
                            cx_list.append(cx)
                            cy_list.append(cy)
                            r_list.append(r)
                bootstrap_ms = (time.perf_counter() - t0) * 1000

                # Weighted Taubin
                t0 = time.perf_counter()
                weights = _compute_gradient_weights(gray, pts)
                t_grad_w = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                if weights is not None:
                    _fit_circle_weighted_taubin(pts, weights)
                t_wtaubin = (time.perf_counter() - t0) * 1000

                # RANSAC
                contour_span = float(np.ptp(pts, axis=0).max())
                adaptive_thresh = max(2.0, contour_span * 0.01)
                t0 = time.perf_counter()
                _ransac_circle(pts, inlier_threshold=adaptive_thresh)
                ransac_ms = (time.perf_counter() - t0) * 1000

                print(f"    frame {fidx:5d}: pts={len(pts):5d} | bootstrap={bootstrap_ms:6.1f}ms (50 Taubin fits) | grad_w={t_grad_w:5.1f}ms | wtaubin={t_wtaubin:5.1f}ms | ransac={ransac_ms:5.1f}ms")
                break

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("Bootstrap: 50 Taubin fits per contour = significant cost")
    print("Weighted Taubin: gradient computation + weighted fit")
    print("RANSAC: 100 iterations of Kaa + residual computation")


if __name__ == "__main__":
    main()

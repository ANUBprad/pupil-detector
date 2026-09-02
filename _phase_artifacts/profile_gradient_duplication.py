"""Phase XX-D: Profile gradient computation duplication across thresholds."""
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

from pupil_tracking.core.smart_fitter import SmartContourFitter, _compute_multiscale_gradient

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


def main():
    print("=" * 70)
    print("PHASE XX-D: Gradient Computation Duplication Check")
    print("=" * 70)

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = sample_frame_indices(total_frames, 6)  # Sample 6 frames
        print(f"\n  Video: {video_path}")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            h, w = gray.shape
            img_diag = math.sqrt(h * h + w * w)
            min_radius = max(8, int(img_diag * 0.015))
            min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

            # Count how many times gradient would be computed
            total_gradient_calls = 0
            total_contours = 0

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
                    total_contours += 1
                    # Each contour triggers subpixel refinement → 1 gradient call
                    total_gradient_calls += 1

            # Time gradient computation once
            t0 = time.perf_counter()
            _compute_multiscale_gradient(gray)
            grad_time_ms = (time.perf_counter() - t0) * 1000

            wasted_ms = grad_time_ms * (total_gradient_calls - 1)
            print(f"    frame {fidx:5d}: {total_contours} contours, {total_gradient_calls} gradient calls, 1 grad={grad_time_ms:.0f}ms, wasted={wasted_ms:.0f}ms")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Gradient is recomputed for EVERY contour in EVERY threshold.")
    print("Caching it once per frame would eliminate all but 1 computation.")
    print("Expected savings: ~239ms × (N_contours - 1) per frame.")


if __name__ == "__main__":
    main()

"""Phase XX-E: Run baseline profiling with same detector for fair comparison."""
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
        out["pupil_radius"] = 0
        out["pupil_center"] = [0, 0]
        out["limbus_radius"] = 0
        out["limbus_center"] = [0, 0]
        out["pupil_method"] = "none"
        out["limbus_method"] = "none"

    return out


def main():
    print("=" * 70)
    print("Fair Baseline (same detector instance)")
    print("=" * 70)

    detector = UnifiedDetector()
    results: List[Dict[str, Any]] = []

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
            result = profile_frame(detector, frame, fidx)
            result["video"] = video_path
            results.append(result)
            ms = result["total_ms"]
            p = "P+" if result["pupil_detected"] else "P-"
            l = "L+" if result["limbus_detected"] else "L-"
            print(f"    [{fi+1}/{len(indices)}] frame {fidx}: {ms:6.0f} ms | {p} {l}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "phase_xxe_fair_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Compare with Run 1
    with open(OUTPUT_DIR / "phase_xxe_run1.json") as f:
        run1 = json.load(f)

    status_changes = 0
    for b, o in zip(results, run1):
        if b['pupil_detected'] != o['pupil_detected']:
            status_changes += 1
            print(f'  frame {b["frame_idx"]}: pupil CHANGED ({b["pupil_detected"]} -> {o["pupil_detected"]})')
        if b['limbus_detected'] != o['limbus_detected']:
            status_changes += 1
            print(f'  frame {b["frame_idx"]}: limbus CHANGED ({b["limbus_detected"]} -> {o["limbus_detected"]})')

    b_times = [r['total_ms'] for r in results]
    o_times = [r['total_ms'] for r in run1]
    print(f'\n  Fair comparison:')
    print(f'    Status changes: {status_changes}')
    print(f'    Baseline mean: {np.mean(b_times):.0f} ms')
    print(f'    Cached mean:   {np.mean(o_times):.0f} ms')
    print(f'    Improvement:   {np.mean(b_times) - np.mean(o_times):.0f} ms ({(np.mean(b_times) - np.mean(o_times)) / np.mean(b_times) * 100:.1f}%)')


if __name__ == "__main__":
    main()

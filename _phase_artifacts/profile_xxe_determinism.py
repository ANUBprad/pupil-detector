"""Phase XX-E: Deterministic correctness comparison using same detector instance."""
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
    """Profile one frame through the full detection pipeline."""
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

    return out


def main():
    print("=" * 70)
    print("PHASE XX-E: Deterministic Correctness (Run 1)")
    print("=" * 70)

    detector = UnifiedDetector()
    run1_results: List[Dict[str, Any]] = []

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
            run1_results.append(result)
            ms = result["total_ms"]
            p = "P+" if result["pupil_detected"] else "P-"
            l = "L+" if result["limbus_detected"] else "L-"
            print(f"    [{fi+1}/{len(indices)}] frame {fidx}: {ms:6.0f} ms | {p} {l}")

    # Run 2: Same detector, same frames (determinism check)
    print("\n" + "=" * 70)
    print("PHASE XX-E: Deterministic Correctness (Run 2)")
    print("=" * 70)

    run2_results: List[Dict[str, Any]] = []

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
            run2_results.append(result)
            ms = result["total_ms"]
            p = "P+" if result["pupil_detected"] else "P-"
            l = "L+" if result["limbus_detected"] else "L-"
            print(f"    [{fi+1}/{len(indices)}] frame {fidx}: {ms:6.0f} ms | {p} {l}")

    # Compare Run 1 vs Run 2
    print("\n" + "=" * 70)
    print("DETERMINISM CHECK (Run 1 vs Run 2)")
    print("=" * 70)

    status_changes = 0
    for r1, r2 in zip(run1_results, run2_results):
        if r1["pupil_detected"] != r2["pupil_detected"]:
            status_changes += 1
            print(f"  frame {r1['frame_idx']}: pupil CHANGED ({r1['pupil_detected']} -> {r2['pupil_detected']})")
        if r1["limbus_detected"] != r2["limbus_detected"]:
            status_changes += 1
            print(f"  frame {r1['frame_idx']}: limbus CHANGED ({r1['limbus_detected']} -> {r2['limbus_detected']})")

    print(f"\n  Determinism: {'PASS' if status_changes == 0 else 'FAIL'} ({status_changes} changes)")

    # Save run1 for comparison
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "phase_xxe_run1.json", "w") as f:
        json.dump(run1_results, f, indent=2, default=str)

    # Performance summary
    times = [r["total_ms"] for r in run1_results]
    print(f"\n  Performance: mean={np.mean(times):.0f}ms  median={np.median(times):.0f}ms  worst={np.max(times):.0f}ms")


if __name__ == "__main__":
    main()

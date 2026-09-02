"""Phase XX-B: Profile real ELITA video processing.

Simple approach: time detect() per frame, extract diagnostics.
NO monkey-patching, NO production code changes.
"""
from __future__ import annotations

import json
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

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
SAMPLE_COUNT = 24
OUTPUT_DIR = Path("_phase_artifacts")


def sample_frame_indices(total: int, count: int) -> List[int]:
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(i * step) for i in range(count)]


def get_video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def read_frame(path: str, idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def profile_frame(detector, frame_bgr: np.ndarray, frame_idx: int) -> Dict[str, Any]:
    """Time detect() and extract diagnostics."""
    t0 = time.perf_counter()
    result = None
    error = None
    try:
        result = detector.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        error = str(e)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "image_shape": list(frame_bgr.shape),
        "total_ms": round(elapsed_ms, 2),
        "error": error,
    }

    if result is not None:
        out["pupil_detected"] = bool(getattr(result.pupil, "detected", False))
        out["limbus_detected"] = bool(getattr(result.limbus, "detected", False))
        if out["pupil_detected"]:
            ell = getattr(result.pupil, "ellipse", None)
            if ell is not None:
                center = getattr(ell, "center", None)
                axes = getattr(ell, "axes", None)
                out["pupil_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center is not None else [0, 0]
                out["pupil_radius"] = round(float(max(axes) if axes else 0), 2)
            else:
                out["pupil_radius"] = 0
                out["pupil_center"] = [0, 0]
        else:
            out["pupil_radius"] = 0
            out["pupil_center"] = [0, 0]
        if out["limbus_detected"]:
            ell = getattr(result.limbus, "ellipse", None)
            if ell is not None:
                center = getattr(ell, "center", None)
                axes = getattr(ell, "axes", None)
                out["limbus_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center is not None else [0, 0]
                out["limbus_radius"] = round(float(max(axes) if axes else 0), 2)
            else:
                out["limbus_radius"] = 0
                out["limbus_center"] = [0, 0]
        else:
            out["limbus_radius"] = 0
            out["limbus_center"] = [0, 0]
        out["confidence"] = round(float(getattr(result, "overall_confidence", 0) or 0), 4)
        ring_status = getattr(result, "ring_status", "unknown")
        out["ring_detected"] = ring_status == "ring_present"
        rr = getattr(result, "ring_radius", None)
        out["ring_radius"] = round(float(rr), 2) if rr is not None else 0
        out["iris_detected"] = bool(getattr(result, "iris_detected", False))
        out["iris_features"] = int(getattr(result, "iris_feature_count", 0) or 0)
        if out["pupil_detected"] and out["limbus_detected"] and out["limbus_radius"] > 0:
            out["pupil_limbus_ratio"] = round(out["pupil_radius"] / out["limbus_radius"], 4)
        else:
            out["pupil_limbus_ratio"] = 0
    else:
        out["pupil_detected"] = False
        out["limbus_detected"] = False
        out["pupil_radius"] = 0
        out["pupil_center"] = [0, 0]
        out["limbus_radius"] = 0
        out["limbus_center"] = [0, 0]
        out["confidence"] = 0
        out["ring_detected"] = False
        out["ring_radius"] = 0
        out["iris_detected"] = False
        out["iris_features"] = 0
        out["pupil_limbus_ratio"] = 0

    return out


def main():
    print("=" * 70)
    print("PHASE XX-B: Real ELITA Video Profiling")
    print("=" * 70)

    from pupil_tracking.core.detector import UnifiedDetector

    print("\nInitializing UnifiedDetector...")
    t_init = time.perf_counter()
    detector = UnifiedDetector()
    init_time = (time.perf_counter() - t_init) * 1000
    print(f"  Detector init: {init_time:.0f} ms")

    all_results: List[Dict[str, Any]] = []

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            print(f"\n  SKIP: {video_path} not found")
            continue

        info = get_video_info(video_path)
        print(f"\n{'=' * 60}")
        print(f"Video: {video_path}")
        print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f} FPS, {info['frame_count']} frames")

        indices = sample_frame_indices(info["frame_count"], SAMPLE_COUNT)
        print(f"  Sampling {len(indices)} frames")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                print(f"  [{fi+1}/{len(indices)}] frame {fidx}: READ FAILED")
                continue

            print(f"  [{fi+1}/{len(indices)}] frame {fidx}...", end=" ", flush=True)
            result = profile_frame(detector, frame, fidx)
            result["video"] = video_path
            all_results.append(result)

            ms = result["total_ms"]
            p_ok = "P+" if result["pupil_detected"] else "P-"
            l_ok = "L+" if result["limbus_detected"] else "L-"
            r_ok = "R+" if result["ring_detected"] else "R-"
            print(f"{ms:8.0f} ms | {p_ok} {l_ok} {r_ok} | conf={result['confidence']:.3f}")

        # Video summary
        vr = [r for r in all_results if r["video"] == video_path]
        p_rate = sum(1 for r in vr if r["pupil_detected"]) / len(vr) * 100
        l_rate = sum(1 for r in vr if r["limbus_detected"]) / len(vr) * 100
        times = [r["total_ms"] for r in vr if r["error"] is None]
        if times:
            print(f"\n  Summary:")
            print(f"    Pupil:   {p_rate:.0f}%")
            print(f"    Limbus:  {l_rate:.0f}%")
            print(f"    Mean:    {np.mean(times):.0f} ms")
            print(f"    Median:  {np.median(times):.0f} ms")
            print(f"    P95:     {np.percentile(times, 95):.0f} ms")
            print(f"    Worst:   {np.max(times):.0f} ms")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxb_profile_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Global summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("GLOBAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total frames: {len(all_results)}")
        p_rate = sum(1 for r in all_results if r["pupil_detected"]) / len(all_results) * 100
        l_rate = sum(1 for r in all_results if r["limbus_detected"]) / len(all_results) * 100
        print(f"Pupil: {p_rate:.0f}%  |  Limbus: {l_rate:.0f}%")

        times = [r["total_ms"] for r in all_results if r["error"] is None]
        if times:
            print(f"\nLatency:")
            print(f"  Mean:   {np.mean(times):.0f} ms")
            print(f"  Median: {np.median(times):.0f} ms")
            print(f"  P95:    {np.percentile(times, 95):.0f} ms")
            print(f"  Worst:  {np.max(times):.0f} ms")

        # Limbus success vs failure timing
        l_ok_times = [r["total_ms"] for r in all_results if r["limbus_detected"]]
        l_fail_times = [r["total_ms"] for r in all_results if not r["limbus_detected"] and r["error"] is None]
        if l_ok_times and l_fail_times:
            print(f"\n  Limbus success timing:  mean={np.mean(l_ok_times):.0f} ms  median={np.median(l_ok_times):.0f} ms")
            print(f"  Limbus failure timing:  mean={np.mean(l_fail_times):.0f} ms  median={np.median(l_fail_times):.0f} ms")

        # Worst 5 frames
        sorted_by_time = sorted(all_results, key=lambda r: r["total_ms"], reverse=True)
        print(f"\n  Worst 5 frames:")
        for r in sorted_by_time[:5]:
            print(f"    frame {r['frame_idx']:5d} ({r['video'][:10]}...)  {r['total_ms']:8.0f} ms  P{'+'if r['pupil_detected'] else '-'} L{'+'if r['limbus_detected'] else '-'} conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()

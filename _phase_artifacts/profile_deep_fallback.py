"""Phase XX-B STEP 4: Deep-dive into the classical fallback bottleneck.

Instruments _classical_limbus and _classical_pupil to measure per-iteration
timing, contour counts, and HoughCircles performance.
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

OUTPUT_DIR = Path("_phase_artifacts")


def read_frame(path: str, idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def profile_frame_deep(detector, frame_bgr: np.ndarray, frame_idx: int) -> Dict[str, Any]:
    """Profile one frame with deep instrumentation of classical fallback."""

    # Save originals
    orig_classical_pupil = detector._classical_pupil
    orig_classical_limbus = detector._classical_limbus
    orig_detect = detector.detect

    classical_pupil_timings: Dict[str, float] = {}
    classical_limbus_timings: Dict[str, float] = {}
    fallback_pupil_used = False
    fallback_limbus_used = False

    def timed_classical_pupil(img):
        nonlocal fallback_pupil_used
        fallback_pupil_used = True
        t_start = time.perf_counter()

        # Instrument internally
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        threshold_times = []
        total_contours = 0
        total_fits = 0

        percentiles = [3, 5, 8, 12, 18, 25, 35]
        for pct in percentiles:
            t_iter = time.perf_counter()
            thresh_val = np.percentile(blurred, pct)
            _, binary = cv2.threshold(blurred, int(thresh_val), 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            total_contours += len(contours)
            total_fits += len(contours)
            threshold_times.append((time.perf_counter() - t_iter) * 1000)

        result = orig_classical_pupil(img)
        elapsed = (time.perf_counter() - t_start) * 1000

        classical_pupil_timings["total_ms"] = round(elapsed, 2)
        classical_pupil_timings["threshold_iterations"] = len(percentiles)
        classical_pupil_timings["total_contours_examined"] = total_contours
        classical_pupil_timings["total_fits_attempted"] = total_fits
        classical_pupil_timings["mean_iteration_ms"] = round(np.mean(threshold_times), 2) if threshold_times else 0
        return result

    def timed_classical_limbus(img, pupil_hint):
        nonlocal fallback_limbus_used
        fallback_limbus_used = True
        t_start = time.perf_counter()

        # Instrument key sub-stages
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        t_edge = time.perf_counter()
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_time = (time.perf_counter() - t_edge) * 1000

        t_hough = time.perf_counter()
        hough_candidates = 0
        hough_params_tried = 0
        for dp, p1, p2 in [(1.5, 80, 40), (2.0, 60, 30)]:
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp,
                                        minDist=50, param1=p1, param2=p2,
                                        minRadius=30, maxRadius=400)
            if circles is not None:
                hough_candidates += len(circles[0])
            hough_params_tried += 1
        hough_time = (time.perf_counter() - t_hough) * 1000

        result = orig_classical_limbus(img, pupil_hint)
        elapsed = (time.perf_counter() - t_start) * 1000

        classical_limbus_timings["total_ms"] = round(elapsed, 2)
        classical_limbus_timings["edge_time_ms"] = round(edge_time, 2)
        classical_limbus_timings["hough_time_ms"] = round(hough_time, 2)
        classical_limbus_timings["hough_params_tried"] = hough_params_tried
        classical_limbus_timings["hough_candidates"] = hough_candidates
        return result

    # Patch
    detector._classical_pupil = timed_classical_pupil
    detector._classical_limbus = timed_classical_limbus

    result = None
    error = None
    t0 = time.perf_counter()
    try:
        result = detector.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        error = str(e)
    total_ms = (time.perf_counter() - t0) * 1000

    # Restore
    detector._classical_pupil = orig_classical_pupil
    detector._classical_limbus = orig_classical_limbus

    out: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "total_ms": round(total_ms, 2),
        "fallback_pupil": fallback_pupil_used,
        "fallback_limbus": fallback_limbus_used,
        "classical_pupil_timings": classical_pupil_timings,
        "classical_limbus_timings": classical_limbus_timings,
        "error": error,
    }

    if result is not None:
        out["pupil_detected"] = bool(getattr(result.pupil, "detected", False))
        out["limbus_detected"] = bool(getattr(result.limbus, "detected", False))
        out["confidence"] = round(float(getattr(result, "overall_confidence", 0) or 0), 4)
        ring_status = getattr(result, "ring_status", "unknown")
        out["ring_detected"] = ring_status == "ring_present"
        ell_p = getattr(result.pupil, "ellipse", None)
        ell_l = getattr(result.limbus, "ellipse", None)
        if ell_p is not None:
            axes = getattr(ell_p, "axes", None)
            out["pupil_radius"] = round(float(max(axes)), 2) if axes else 0
        else:
            out["pupil_radius"] = 0
        if ell_l is not None:
            axes = getattr(ell_l, "axes", None)
            out["limbus_radius"] = round(float(max(axes)), 2) if axes else 0
        else:
            out["limbus_radius"] = 0
    else:
        out["pupil_detected"] = False
        out["limbus_detected"] = False
        out["confidence"] = 0
        out["ring_detected"] = False
        out["pupil_radius"] = 0
        out["limbus_radius"] = 0

    return out


def main():
    # Load previous profile results to find the worst frames
    prev_path = OUTPUT_DIR / "phase_xxb_profile_results.json"
    if not prev_path.exists():
        print("Run profile_elita_videos.py first")
        return

    with open(prev_path) as f:
        prev_results = json.load(f)

    # Find worst 10 frames + 5 best frames for comparison
    sorted_by_time = sorted(prev_results, key=lambda r: r["total_ms"], reverse=True)
    worst_frames = sorted_by_time[:10]
    best_frames = sorted_by_time[-5:]

    test_frames = []
    for r in worst_frames:
        test_frames.append((r["video"], r["frame_idx"], "worst"))
    for r in best_frames:
        test_frames.append((r["video"], r["frame_idx"], "best"))

    print("=" * 70)
    print("PHASE XX-B STEP 4: Deep-Dive Classical Fallback Profiling")
    print("=" * 70)
    print(f"\nProfiling {len(test_frames)} frames (10 worst + 5 best)")

    from pupil_tracking.core.detector import UnifiedDetector
    detector = UnifiedDetector()

    all_deep: List[Dict[str, Any]] = []

    for video_path, fidx, category in test_frames:
        if not os.path.exists(video_path):
            continue
        frame = read_frame(video_path, fidx)
        if frame is None:
            continue

        print(f"\n  [{category:5s}] frame {fidx} ({video_path[:15]}...)...", end=" ", flush=True)
        deep = profile_frame_deep(detector, frame, fidx)
        deep["video"] = video_path
        deep["category"] = category
        all_deep.append(deep)

        ms = deep["total_ms"]
        pupil = "P+" if deep["pupil_detected"] else "P-"
        limbus = "L+" if deep["limbus_detected"] else "L-"
        fb_p = " FB_P" if deep["fallback_pupil"] else ""
        fb_l = " FB_L" if deep["fallback_limbus"] else ""
        print(f"{ms:8.0f} ms | {pupil} {limbus}{fb_p}{fb_l}")

        if deep["fallback_limbus"] and deep["classical_limbus_timings"]:
            ct = deep["classical_limbus_timings"]
            print(f"           classical_limbus: {ct.get('total_ms', 0):.0f} ms "
                  f"| edge: {ct.get('edge_time_ms', 0):.0f} ms "
                  f"| hough: {ct.get('hough_time_ms', 0):.0f} ms "
                  f"| candidates: {ct.get('hough_candidates', 0)}")
        if deep["fallback_pupil"] and deep["classical_pupil_timings"]:
            pt = deep["classical_pupil_timings"]
            print(f"           classical_pupil: {pt.get('total_ms', 0):.0f} ms "
                  f"| iterations: {pt.get('threshold_iterations', 0)} "
                  f"| contours: {pt.get('total_contours_examined', 0)} "
                  f"| fits: {pt.get('total_fits_attempted', 0)}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxb_deep_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_deep, f, indent=2, default=str)
    print(f"\nDeep profile saved to {out_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("DEEP PROFILE SUMMARY")
    print(f"{'=' * 70}")
    for cat in ["worst", "best"]:
        frames = [r for r in all_deep if r["category"] == cat]
        if not frames:
            continue
        times = [r["total_ms"] for r in frames]
        fb_limbus = [r for r in frames if r["fallback_limbus"]]
        fb_pupil = [r for r in frames if r["fallback_pupil"]]
        print(f"\n  {cat.upper()} frames ({len(frames)}):")
        print(f"    Total time:  mean={np.mean(times):.0f} ms  range={min(times):.0f}-{max(times):.0f} ms")
        print(f"    Fallbacks:   limbus={len(fb_limbus)}/{len(frames)}  pupil={len(fb_pupil)}/{len(frames)}")
        if fb_limbus:
            lt = [r["classical_limbus_timings"]["total_ms"] for r in fb_limbus if r["classical_limbus_timings"]]
            if lt:
                print(f"    Classical limbus time: mean={np.mean(lt):.0f} ms  range={min(lt):.0f}-{max(lt):.0f} ms")


if __name__ == "__main__":
    main()

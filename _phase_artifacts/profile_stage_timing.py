"""Phase XX-B STEPS 6-7: Compare success vs failure + check repeated work.

Instrument detect() to capture per-stage timing when classical fallback
is triggered.
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


def profile_with_stage_timing(detector, frame_bgr: np.ndarray, frame_idx: int) -> Dict[str, Any]:
    """Profile with per-stage timing using method wrappers."""
    timings: Dict[str, float] = {}

    # Save originals
    orig_ring = detector._detect_ring
    orig_ring_preprocess = detector._ring_preprocessor.preprocess
    orig_ml = detector.ml_engine.detect
    orig_extract = detector._extract_structure
    orig_classical_pupil = detector._classical_pupil
    orig_classical_limbus = detector._classical_limbus
    orig_cross = detector._cross_validate_and_reject

    def wrap(name, fn):
        def timed(*args, **kwargs):
            t = time.perf_counter()
            result = fn(*args, **kwargs)
            timings[name] = timings.get(name, 0) + (time.perf_counter() - t) * 1000
            return result
        return timed

    detector._detect_ring = wrap("ring_detection", orig_ring)
    detector._ring_preprocessor.preprocess = wrap("preprocessing", orig_ring_preprocess)
    detector.ml_engine.detect = wrap("ml_segmentation", orig_ml)
    detector._extract_structure = wrap("structure_extraction", orig_extract)
    detector._classical_pupil = wrap("classical_pupil", orig_classical_pupil)
    detector._classical_limbus = wrap("classical_limbus", orig_classical_limbus)
    detector._cross_validate_and_reject = wrap("cross_validation", orig_cross)

    t0 = time.perf_counter()
    result = None
    error = None
    try:
        result = detector.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        error = str(e)
    timings["total"] = (time.perf_counter() - t0) * 1000

    # Restore
    detector._detect_ring = orig_ring
    detector._ring_preprocessor.preprocess = orig_ring_preprocess
    detector.ml_engine.detect = orig_ml
    detector._extract_structure = orig_extract
    detector._classical_pupil = orig_classical_pupil
    detector._classical_limbus = orig_classical_limbus
    detector._cross_validate_and_reject = orig_cross

    out: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "error": error,
    }

    if result is not None:
        out["pupil_detected"] = bool(getattr(result.pupil, "detected", False))
        out["limbus_detected"] = bool(getattr(result.limbus, "detected", False))
        out["confidence"] = round(float(getattr(result, "overall_confidence", 0) or 0), 4)
        out["pupil_method"] = str(getattr(result.pupil, "method", "none"))
        out["limbus_method"] = str(getattr(result.limbus, "method", "none"))
    else:
        out["pupil_detected"] = False
        out["limbus_detected"] = False
        out["confidence"] = 0
        out["pupil_method"] = "none"
        out["limbus_method"] = "none"

    return out


def main():
    prev_path = OUTPUT_DIR / "phase_xxb_profile_results.json"
    with open(prev_path) as f:
        prev_results = json.load(f)

    # Get worst frames (likely classical fallback) + best frames
    sorted_r = sorted(prev_results, key=lambda r: r["total_ms"], reverse=True)
    test_frames = []
    for r in sorted_r[:12]:
        test_frames.append((r["video"], r["frame_idx"], "slow"))
    for r in sorted_r[-6:]:
        test_frames.append((r["video"], r["frame_idx"], "fast"))

    print("=" * 70)
    print("PHASE XX-B STEPS 6-7: Stage Timing + Repeated Work Analysis")
    print("=" * 70)

    from pupil_tracking.core.detector import UnifiedDetector
    detector = UnifiedDetector()

    all_results: List[Dict[str, Any]] = []

    for video_path, fidx, cat in test_frames:
        if not os.path.exists(video_path):
            continue
        frame = read_frame(video_path, fidx)
        if frame is None:
            continue

        print(f"\n  [{cat:4s}] frame {fidx}...", end=" ", flush=True)
        r = profile_with_stage_timing(detector, frame, fidx)
        r["video"] = video_path
        r["category"] = cat
        all_results.append(r)

        t = r["timings_ms"]
        total = t.get("total", 0)
        p = "P+" if r["pupil_detected"] else "P-"
        l = "L+" if r["limbus_detected"] else "L-"
        fb_p = t.get("classical_pupil", 0)
        fb_l = t.get("classical_limbus", 0)
        fb_str = ""
        if fb_p > 0:
            fb_str += f" FB_pupil={fb_p:.0f}"
        if fb_l > 0:
            fb_str += f" FB_limbus={fb_l:.0f}"
        print(f"{total:7.0f} ms | {p} {l} | ml={t.get('ml_segmentation',0):.0f} fit={t.get('structure_extraction',0):.0f}{fb_str}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxb_stage_timing.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 70}")
    print("STAGE TIMING SUMMARY")
    print(f"{'=' * 70}")

    stages = ["ring_detection", "preprocessing", "ml_segmentation",
              "structure_extraction", "classical_pupil", "classical_limbus",
              "cross_validation", "total"]

    for cat in ["slow", "fast"]:
        frames = [r for r in all_results if r["category"] == cat]
        if not frames:
            continue
        print(f"\n  {cat.upper()} frames ({len(frames)}):")
        for s in stages:
            vals = [r["timings_ms"].get(s, 0) for r in frames]
            nonzero = [v for v in vals if v > 0]
            if nonzero:
                print(f"    {s:30s}: mean={np.mean(nonzero):7.0f} ms  n={len(nonzero)}/{len(vals)}")
            else:
                print(f"    {s:30s}: never triggered")

    # Repeated work analysis
    print(f"\n{'=' * 70}")
    print("REPEATED WORK ANALYSIS")
    print(f"{'=' * 70}")

    # Check if ring detection, preprocessing, ML are called once or multiple times
    for r in all_results:
        t = r["timings_ms"]
        ml = t.get("ml_segmentation", 0)
        ring = t.get("ring_detection", 0)
        prep = t.get("preprocessing", 0)
        fit = t.get("structure_extraction", 0)
        fb_p = t.get("classical_pupil", 0)
        fb_l = t.get("classical_limbus", 0)
        total = t.get("total", 0)

        # Check for overhead (total - sum of parts)
        parts_sum = ring + prep + ml + fit + fb_p + fb_l
        overhead = total - parts_sum
        if overhead > 500:
            print(f"  frame {r['frame_idx']:5d}: overhead={overhead:.0f} ms (total={total:.0f} - parts={parts_sum:.0f})")

    # Compare slow vs fast stage breakdown
    slow_frames = [r for r in all_results if r["category"] == "slow"]
    fast_frames = [r for r in all_results if r["category"] == "fast"]
    if slow_frames and fast_frames:
        print(f"\n  Stage comparison (slow vs fast):")
        for s in stages:
            slow_vals = [r["timings_ms"].get(s, 0) for r in slow_frames]
            fast_vals = [r["timings_ms"].get(s, 0) for r in fast_frames]
            slow_mean = np.mean([v for v in slow_vals if v > 0]) if any(v > 0 for v in slow_vals) else 0
            fast_mean = np.mean([v for v in fast_vals if v > 0]) if any(v > 0 for v in fast_vals) else 0
            if slow_mean > 0 or fast_mean > 0:
                ratio = slow_mean / max(fast_mean, 0.01)
                print(f"    {s:30s}: slow={slow_mean:7.0f} ms  fast={fast_mean:7.0f} ms  ratio={ratio:.1f}x")


if __name__ == "__main__":
    main()

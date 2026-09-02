"""Phase XX-B STEP 5: Investigate limbus failures.

Analyze frames where limbus fails vs succeeds. Compare image statistics,
ML mask quality, and classical fallback behavior.
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


def compute_image_stats(frame: np.ndarray) -> Dict[str, float]:
    """Compute basic image quality metrics."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Brightness
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Contrast (Michelson-like)
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float((p95 - p5) / max(p95 + p5, 1))

    # Blur metric (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_metric = float(np.var(laplacian))

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (h * w))

    # Center region (where eye typically is)
    ch, cw = h // 4, w // 4
    center_region = gray[ch:3*ch, cw:3*cw]
    center_brightness = float(np.mean(center_region))
    center_contrast = float(np.std(center_region))

    return {
        "mean_brightness": round(mean_brightness, 2),
        "std_brightness": round(std_brightness, 2),
        "contrast": round(contrast, 4),
        "blur_metric": round(blur_metric, 2),
        "edge_density": round(edge_density, 6),
        "center_brightness": round(center_brightness, 2),
        "center_contrast": round(center_contrast, 2),
        "width": w,
        "height": h,
    }


def analyze_frame(detector, frame_bgr: np.ndarray, frame_idx: int) -> Dict[str, Any]:
    """Analyze one frame deeply — ML masks, fitting, classical fallback."""
    result = None
    ml_time = 0
    fit_time = 0
    classical_time = 0
    error = None

    # Time ML
    t_ml = time.perf_counter()
    try:
        ml_result = detector.ml_engine.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        ml_result = None
        error = str(e)
    ml_time = (time.perf_counter() - t_ml) * 1000

    # Analyze ML masks
    ml_mask_info = {}
    if ml_result is not None:
        raw_mask = getattr(ml_result, "_raw_mask", None)
        if raw_mask is None:
            raw_mask = getattr(ml_result, "raw_mask", None)
        if raw_mask is not None:
            ml_mask_info["mask_shape"] = list(raw_mask.shape)
            ml_mask_info["mask_dtype"] = str(raw_mask.dtype)
            ml_mask_info["unique_values"] = sorted(int(v) for v in np.unique(raw_mask))
            for cls_val, cls_name in [(1, "pupil"), (2, "iris"), (3, "ring")]:
                cls_mask = (raw_mask == cls_val).astype(np.uint8)
                pixel_count = int(np.sum(cls_mask))
                ml_mask_info[f"{cls_name}_pixels"] = pixel_count
                if pixel_count > 0:
                    contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    ml_mask_info[f"{cls_name}_contours"] = len(contours)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        ml_mask_info[f"{cls_name}_largest_area"] = int(cv2.contourArea(largest))
                else:
                    ml_mask_info[f"{cls_name}_contours"] = 0
                    ml_mask_info[f"{cls_name}_largest_area"] = 0

    # Full detect with timing
    t0 = time.perf_counter()
    try:
        result = detector.detect(frame_bgr, frame_number=frame_idx, source="video")
    except Exception as e:
        result = None
        error = str(e)
    total_ms = (time.perf_counter() - t0) * 1000

    # Extract diagnostics
    out: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "total_ms": round(total_ms, 2),
        "ml_time_ms": round(ml_time, 2),
        "ml_mask_info": ml_mask_info,
        "image_stats": compute_image_stats(frame_bgr),
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
            center = getattr(ell_p, "center", None)
            out["pupil_radius"] = round(float(max(axes)), 2) if axes else 0
            out["pupil_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center else [0, 0]
        else:
            out["pupil_radius"] = 0
            out["pupil_center"] = [0, 0]
        if ell_l is not None:
            axes = getattr(ell_l, "axes", None)
            center = getattr(ell_l, "center", None)
            out["limbus_radius"] = round(float(max(axes)), 2) if axes else 0
            out["limbus_center"] = [round(float(center[0]), 2), round(float(center[1]), 2)] if center else [0, 0]
        else:
            out["limbus_radius"] = 0
            out["limbus_center"] = [0, 0]

        # Pupil/Limbus method
        out["pupil_method"] = str(getattr(result.pupil, "method", "unknown"))
        out["limbus_method"] = str(getattr(result.limbus, "method", "unknown"))
    else:
        out["pupil_detected"] = False
        out["limbus_detected"] = False
        out["confidence"] = 0
        out["ring_detected"] = False
        out["pupil_radius"] = 0
        out["pupil_center"] = [0, 0]
        out["limbus_radius"] = 0
        out["limbus_center"] = [0, 0]
        out["pupil_method"] = "none"
        out["limbus_method"] = "none"

    return out


def main():
    # Load previous results
    prev_path = OUTPUT_DIR / "phase_xxb_profile_results.json"
    if not prev_path.exists():
        print("Run profile_elita_videos.py first")
        return

    with open(prev_path) as f:
        prev_results = json.load(f)

    # Find limbus success and failure frames
    l_success = [r for r in prev_results if r["limbus_detected"]]
    l_failure = [r for r in prev_results if not r["limbus_detected"]]

    # Sample 8 success + 8 failure for comparison
    test_frames = []
    for r in l_success[:8]:
        test_frames.append((r["video"], r["frame_idx"], "success"))
    for r in l_failure[:8]:
        test_frames.append((r["video"], r["frame_idx"], "failure"))

    print("=" * 70)
    print("PHASE XX-B STEP 5: Limbus Failure Investigation")
    print("=" * 70)
    print(f"\nAnalyzing {len(test_frames)} frames ({len(l_success)} success, {len(l_failure)} failure available)")

    from pupil_tracking.core.detector import UnifiedDetector
    detector = UnifiedDetector()

    all_analysis: List[Dict[str, Any]] = []

    for video_path, fidx, category in test_frames:
        if not os.path.exists(video_path):
            continue
        frame = read_frame(video_path, fidx)
        if frame is None:
            continue

        print(f"\n  [{category:7s}] frame {fidx} ({video_path[:15]}...)...", end=" ", flush=True)
        analysis = analyze_frame(detector, frame, fidx)
        analysis["video"] = video_path
        analysis["category"] = category
        all_analysis.append(analysis)

        ms = analysis["total_ms"]
        pupil = "P+" if analysis["pupil_detected"] else "P-"
        limbus = "L+" if analysis["limbus_detected"] else "L-"
        stats = analysis["image_stats"]
        print(f"{ms:7.0f} ms | {pupil} {limbus} | bright={stats['mean_brightness']:.0f} blur={stats['blur_metric']:.0f} edge={stats['edge_density']:.4f}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxb_failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {out_path}")

    # Comparison
    print(f"\n{'=' * 70}")
    print("SUCCESS vs FAILURE COMPARISON")
    print(f"{'=' * 70}")

    for cat in ["success", "failure"]:
        frames = [r for r in all_analysis if r["category"] == cat]
        if not frames:
            continue
        print(f"\n  {cat.upper()} frames ({len(frames)}):")
        for key in ["mean_brightness", "std_brightness", "contrast", "blur_metric", "edge_density", "center_brightness", "center_contrast"]:
            vals = [r["image_stats"][key] for r in frames]
            print(f"    {key:25s}: mean={np.mean(vals):8.2f}  std={np.std(vals):8.2f}  range=[{min(vals):.2f}, {max(vals):.2f}]")

        # ML mask comparison
        pupil_pixels = [r["ml_mask_info"].get("pupil_pixels", 0) for r in frames]
        iris_pixels = [r["ml_mask_info"].get("iris_pixels", 0) for r in frames]
        pupil_contours = [r["ml_mask_info"].get("pupil_contours", 0) for r in frames]
        iris_contours = [r["ml_mask_info"].get("iris_contours", 0) for r in frames]
        print(f"    {'pupil_ml_pixels':25s}: mean={np.mean(pupil_pixels):8.0f}  range=[{min(pupil_pixels)}, {max(pupil_pixels)}]")
        print(f"    {'iris_ml_pixels':25s}: mean={np.mean(iris_pixels):8.0f}  range=[{min(iris_pixels)}, {max(iris_pixels)}]")
        print(f"    {'pupil_ml_contours':25s}: mean={np.mean(pupil_contours):8.1f}  range=[{min(pupil_contours)}, {max(pupil_contours)}]")
        print(f"    {'iris_ml_contours':25s}: mean={np.mean(iris_contours):8.1f}  range=[{min(iris_contours)}, {max(iris_contours)}]")

        # Timing
        times = [r["total_ms"] for r in frames]
        ml_times = [r["ml_time_ms"] for r in frames]
        print(f"    {'total_time_ms':25s}: mean={np.mean(times):8.0f}  range=[{min(times):.0f}, {max(times):.0f}]")
        print(f"    {'ml_time_ms':25s}: mean={np.mean(ml_times):8.0f}  range=[{min(ml_times):.0f}, {max(ml_times):.0f}]")


if __name__ == "__main__":
    main()

"""Phase XX-D: Check for repeated contour work across thresholds.

Determines whether the same contours are being refined multiple times
across different threshold iterations.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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


def contour_fingerprint(cnt: np.ndarray) -> str:
    """Create a fingerprint for a contour based on its bounding box and area."""
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    # Use bounding box + area as a fast fingerprint
    return f"{x},{y},{w},{h},{int(area)}"


def analyze_frame(frame_bgr, frame_idx):
    """Analyze one frame to find repeated contours across thresholds."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_diag = math.sqrt(h * h + w * w)
    min_radius = max(8, int(img_diag * 0.015))
    max_radius = int(img_diag * 0.25)
    min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    all_fingerprints: List[Tuple[int, str, float, int]] = []  # (pct, fp, area, len)
    all_contours: List[Tuple[int, np.ndarray, float, int]] = []

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
            fp = contour_fingerprint(cnt)
            all_fingerprints.append((pct, fp, area, len(cnt)))
            all_contours.append((pct, cnt, area, len(cnt)))

    # Check for duplicate fingerprints
    fp_counts: Dict[str, List[int]] = {}
    for pct, fp, area, length in all_fingerprints:
        fp_counts.setdefault(fp, []).append(pct)

    duplicates = {fp: pcts for fp, pcts in fp_counts.items() if len(pcts) > 1}

    # Check contour overlap using IoU
    iou_duplicates = 0
    iou_total_pairs = 0
    for i in range(len(all_contours)):
        for j in range(i + 1, len(all_contours)):
            pct_i, cnt_i, _, _ = all_contours[i]
            pct_j, cnt_j, _, _ = all_contours[j]
            if pct_i == pct_j:
                continue  # Same threshold

            # Create masks
            mask_i = np.zeros((h, w), dtype=np.uint8)
            mask_j = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_i, [cnt_i], -1, 1, -1)
            cv2.drawContours(mask_j, [cnt_j], -1, 1, -1)

            intersection = np.sum(mask_i & mask_j)
            union = np.sum(mask_i | mask_j)
            iou = intersection / union if union > 0 else 0

            iou_total_pairs += 1
            if iou > 0.5:
                iou_duplicates += 1

    return {
        "frame_idx": frame_idx,
        "total_contours": len(all_fingerprints),
        "unique_fingerprints": len(fp_counts),
        "duplicate_fingerprints": len(duplicates),
        "contours_per_threshold": {
            pct: sum(1 for p, _, _, _ in all_fingerprints if p == pct)
            for pct in [3, 5, 8, 12, 18, 25]
        },
        "iou_duplicate_pairs": iou_duplicates,
        "iou_total_pairs": iou_total_pairs,
        "iou_duplicate_ratio": round(iou_duplicates / iou_total_pairs, 4) if iou_total_pairs > 0 else 0,
    }


def main():
    print("=" * 70)
    print("PHASE XX-D: Cross-Threshold Contour Overlap Analysis")
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

            result = analyze_frame(frame, fidx)
            result["video"] = video_path
            all_results.append(result)

            dup = result["duplicate_fingerprints"]
            iou_dup = result["iou_duplicate_pairs"]
            iou_total = result["iou_total_pairs"]
            print(f"    frame {fidx:5d}: {result['total_contours']:3d} contours, {result['unique_fingerprints']:3d} unique, {dup:3d} fp-duplicates, {iou_dup}/{iou_total} iou-duplicates")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_contours = sum(r["total_contours"] for r in all_results)
    total_unique = sum(r["unique_fingerprints"] for r in all_results)
    total_fp_dup = sum(r["duplicate_fingerprints"] for r in all_results)
    total_iou_dup = sum(r["iou_duplicate_pairs"] for r in all_results)
    total_iou_pairs = sum(r["iou_total_pairs"] for r in all_results)

    print(f"Total contours across all frames: {total_contours}")
    print(f"Unique fingerprints: {total_unique}")
    print(f"Fingerprint duplicates (same contour across thresholds): {total_fp_dup}")
    print(f"IoU duplicates (IoU > 0.5): {total_iou_dup}/{total_iou_pairs}")
    print(f"IoU duplicate ratio: {total_iou_dup/total_iou_pairs:.2%}" if total_iou_pairs > 0 else "No pairs")

    # Per-threshold distribution
    print("\nContours per threshold (mean across frames):")
    for pct in [3, 5, 8, 12, 18, 25]:
        counts = [r["contours_per_threshold"].get(pct, 0) for r in all_results]
        print(f"  pct={pct:2d}: mean={np.mean(counts):.1f}  median={np.median(counts):.0f}  max={np.max(counts)}")

    # Save
    out_path = Path("_phase_artifacts/phase_xxd_overlap_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

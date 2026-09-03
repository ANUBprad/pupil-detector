"""Phase XX-H: Profile the full ONNX wrapper detect() method."""
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

from pupil_tracking.ml.onnx_inference import ONNXInference
from pupil_tracking.ml.inference import SegmentationInference

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


def profile_wrapper_detect(wrapper, frame_bgr, frame_idx):
    """Profile each stage of the ONNX wrapper's detect() method."""
    stages = {}

    # Stage 1: Ring masker
    t0 = time.perf_counter()
    clean_bgr = frame_bgr
    roi_mask = None
    if wrapper._ring_masker is not None:
        try:
            clean_bgr, marker_mask, ring_result = wrapper._ring_masker.remove_with_diagnostics(clean_bgr)
            if getattr(ring_result, "ring_centre", None) is not None:
                cx, cy = int(round(ring_result.ring_centre[0])), int(round(ring_result.ring_centre[1]))
                inner_r = int(round(ring_result.ring_inner_radius)) if getattr(ring_result, "ring_inner_radius", None) is not None else None
                if inner_r is not None and inner_r > 4:
                    h, w = clean_bgr.shape[:2]
                    roi_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(roi_mask, (cx, cy), inner_r, 255, -1)
        except Exception:
            clean_bgr, _ = wrapper._ring_masker.remove(clean_bgr)
    stages['ring_masker'] = (time.perf_counter() - t0) * 1000

    # Stage 2: Reflection remover
    t0 = time.perf_counter()
    if wrapper._reflection_remover is not None:
        clean_bgr, _ = wrapper._reflection_remover.remove(clean_bgr, roi_mask=roi_mask)
    stages['reflection_remover'] = (time.perf_counter() - t0) * 1000

    # Stage 3: Red light filter
    t0 = time.perf_counter()
    if wrapper._red_light_enabled:
        if wrapper._red_light_filter is None:
            wrapper._red_light_filter = wrapper._get_red_light_filter()
        if wrapper._red_light_filter is not None:
            clean_bgr, _ = wrapper._red_light_filter.apply(
                clean_bgr, frame_number=frame_idx
            )
    stages['red_light_filter'] = (time.perf_counter() - t0) * 1000

    # Stage 4: ONNX inference
    t0 = time.perf_counter()
    masks = wrapper._engine.infer(clean_bgr)
    stages['onnx_inference'] = (time.perf_counter() - t0) * 1000

    # Stage 5: Mask processing
    t0 = time.perf_counter()
    # Build integer label mask
    h, w = clean_bgr.shape[:2]
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    if masks.get("pupil") is not None:
        pupil_resized = cv2.resize(masks["pupil"], (w, h), interpolation=cv2.INTER_NEAREST)
        raw_mask[pupil_resized > 127] = 1
    if masks.get("iris") is not None:
        iris_resized = cv2.resize(masks["iris"], (w, h), interpolation=cv2.INTER_NEAREST)
        raw_mask[iris_resized > 127] = 2
    stages['mask_processing'] = (time.perf_counter() - t0) * 1000

    stages['total_wrapper'] = sum(stages.values())

    return stages


def main():
    print("=" * 70)
    print("PHASE XX-H: Full Wrapper detect() Profiling")
    print("=" * 70)

    # Load the ONNX wrapper (same as detector uses)
    from pupil_tracking.core.detector import UnifiedDetector
    detector = UnifiedDetector()
    wrapper = detector.ml_engine

    print(f"  Wrapper type: {type(wrapper).__name__}")
    print(f"  Engine type: {type(wrapper._engine).__name__}")
    print(f"  Available: {wrapper.available}")

    all_results: List[Dict[str, Any]] = []

    for video_path in VIDEO_FILES:
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = sample_frame_indices(total_frames, SAMPLE_COUNT)
        print(f"\n  Video: {video_path} ({total_frames} frames)")

        for fi, fidx in enumerate(indices):
            frame = read_frame(video_path, fidx)
            if frame is None:
                continue

            stages = profile_wrapper_detect(wrapper, frame, fidx)

            result = {
                "frame_idx": fidx,
                "video": video_path,
                "stages": {k: round(v, 2) for k, v in stages.items()},
            }
            all_results.append(result)

            ms = stages['total_wrapper']
            print(f"    [{fi+1:2d}/{len(indices)}] frame {fidx:5d}: {ms:6.1f}ms | ring={stages['ring_masker']:.1f} refl={stages['reflection_remover']:.1f} red={stages['red_light_filter']:.1f} onnx={stages['onnx_inference']:.1f} mask={stages['mask_processing']:.1f}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxh_wrapper_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrapper profile saved to {out_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("WRAPPER STAGE BREAKDOWN (all frames)")
    print(f"{'=' * 70}")

    stages_data: Dict[str, List[float]] = {}
    for r in all_results:
        for stage, ms in r["stages"].items():
            stages_data.setdefault(stage, []).append(ms)

    total_wrapper = sum(sum(v) for v in stages_data.values())
    print(f"\n  {'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s} {'Total':>10s} {'%':>6s}")
    print(f"  {'-'*75}")
    for stage in sorted(stages_data, key=lambda s: -sum(stages_data[s])):
        vals = stages_data[stage]
        total = sum(vals)
        pct = total / total_wrapper * 100 if total_wrapper > 0 else 0
        print(f"  {stage:25s} {np.mean(vals):10.2f} {np.median(vals):10.2f} {np.percentile(vals, 95):10.2f} {np.max(vals):10.2f} {total:10.1f} {pct:5.1f}%")


if __name__ == "__main__":
    main()

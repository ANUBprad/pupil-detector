"""Phase 12 - Profile individual stages of detect() to find bottleneck.

Instruments _ONNXEngineWrapper.detect and UnifiedDetector.detect
to measure time spent in each sub-stage.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PUPIL_TRACKING_SILENT"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    from pupil_tracking.utils.config import get_config
    from pupil_tracking.core.detector import UnifiedDetector, _ONNXEngineWrapper
    import cv2
    import numpy as np

    cfg = get_config()
    det = UnifiedDetector(config=cfg)
    wrapper = det.ml_engine

    img = np.full((1080, 1920, 3), 30, np.uint8)
    cv2.circle(img, (960, 540), 420, 90, -1)
    cv2.circle(img, (960, 540), 200, 10, -1)

    print("=" * 66)
    print("DETECT() STAGE PROFILING")
    print("=" * 66)

    # --- Profile ONNX engine detect (preprocessing + inference) ---
    print("\n-- _ONNXEngineWrapper.detect() sub-stages --")

    # Monkey-patch to instrument
    orig_remove_ring = wrapper._ring_masker.remove_with_diagnostics
    orig_remove_reflection = wrapper._reflection_remover.remove
    orig_infer = wrapper._engine.infer

    timings = {}

    def timed_ring_mask(image):
        t0 = time.perf_counter()
        out = orig_remove_ring(image)
        timings["ring_mask"] = (time.perf_counter() - t0) * 1000.0
        return out

    def timed_reflection(image, roi_mask=None):
        t0 = time.perf_counter()
        out = orig_remove_reflection(image, roi_mask=roi_mask)
        timings["reflection"] = (time.perf_counter() - t0) * 1000.0
        return out

    def timed_infer(image):
        t0 = time.perf_counter()
        out = orig_infer(image)
        timings["onnx_infer"] = (time.perf_counter() - t0) * 1000.0
        return out

    wrapper._ring_masker.remove_with_diagnostics = timed_ring_mask
    wrapper._reflection_remover.remove = timed_reflection
    wrapper._engine.infer = timed_infer

    # --- Profile UnifiedDetector.detect() outer stages ---
    print("\n-- UnifiedDetector.detect() sub-stages --")

    orig_detect_ring = det._detect_ring
    orig_preprocess = det._ring_preprocessor.preprocess
    orig_extract = det._extract_structure
    orig_classical_pupil = det._classical_pupil
    orig_classical_limbus = det._classical_limbus

    outer_timings = {}

    def timed_detect_ring(image, force_mode=None):
        t0 = time.perf_counter()
        out = orig_detect_ring(image, force_mode)
        outer_timings["ring_detect"] = (time.perf_counter() - t0) * 1000.0
        return out

    def timed_preprocess(image, ring_result):
        t0 = time.perf_counter()
        out = orig_preprocess(image, ring_result)
        outer_timings["preprocess"] = (time.perf_counter() - t0) * 1000.0
        return out

    def timed_extract(mask, gray, ring_result=None):
        t0 = time.perf_counter()
        out = orig_extract(mask, gray, ring_result=ring_result)
        outer_timings["extract_fit"] = (time.perf_counter() - t0) * 1000.0
        return out

    det._detect_ring = timed_detect_ring
    det._ring_preprocessor.preprocess = timed_preprocess
    det._extract_structure = timed_extract

    # Run detect and collect timings
    total_t0 = time.perf_counter()
    result = det.detect(img, source="profile-stages")
    total_ms = (time.perf_counter() - total_t0) * 1000.0

    # Print results
    all_timings = {}
    for k, v in sorted(outer_timings.items()):
        print(f"  {k:<30} {v:8.1f} ms")
        all_timings[k] = v
    for k, v in sorted(timings.items()):
        print(f"  {k:<30} {v:8.1f} ms")
        all_timings[k] = v

    print(f"  {'TOTAL detect()':<30} {total_ms:8.1f} ms")
    all_timings["TOTAL"] = total_ms

    accounted = sum(all_timings.values()) - total_ms
    print(f"  {'(accounted sum)':<30} {sum(v for k,v in all_timings.items() if k != 'TOTAL'):8.1f} ms")

    print("\n-- Result quality --")
    print(f"  pupil detected:   {result.pupil.detected}")
    print(f"  limbus detected:  {result.limbus.detected}")
    print(f"  iris features:    {getattr(result, 'iris_feature_count', 'N/A')}")

    # Restore
    wrapper._ring_masker.remove_with_diagnostics = orig_remove_ring
    wrapper._reflection_remover.remove = orig_remove_reflection
    wrapper._engine.infer = orig_infer
    det._detect_ring = orig_detect_ring
    det._ring_preprocessor.preprocess = orig_preprocess
    det._extract_structure = orig_extract

    # Save
    out = Path(__file__).parent / "phase12_stage_profile.json"
    with open(out, "w") as f:
        json.dump(all_timings, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

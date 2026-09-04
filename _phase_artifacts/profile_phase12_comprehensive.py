"""Phase 12 - Comprehensive startup profile (headless-safe).

Measures every major stage of startup and detection cost:
  A. Module imports (torch, onnxruntime, cv2, numpy)
  B. UnifiedDetector construction (ONNX model load)
  C. _init_detector sub-stages
  D. First + warm detect() on 1920x1080 synthetic
  E. Video-frame latency (classic loop, 10 frames)
  F. Fast-pipeline lazy import cost (torch deferred)
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


def ms(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"  {label:<52} {dt:8.1f} ms")
    return out, dt


def main():
    print("=" * 66)
    print("PHASE 12 - COMPREHENSIVE STARTUP PROFILE (BEFORE OPTIMISATION)")
    print("=" * 66)
    R = {}
    t_all = time.perf_counter()

    # -- A. Module imports --
    print("\n-- A. Module imports (fresh interpreter) --")
    _, R["A_torch"] = ms("import torch", lambda: __import__("torch"))
    _, R["A_onnxrt"] = ms("import onnxruntime", lambda: __import__("onnxruntime"))
    _, R["A_cv2"] = ms("import cv2", lambda: __import__("cv2"))
    _, R["A_np"] = ms("import numpy", lambda: __import__("numpy"))

    from pupil_tracking.utils.config import get_config
    cfg = get_config()

    # -- B. UnifiedDetector construction --
    print("\n-- B. UnifiedDetector.__init__() (ONNX model load) --")
    from pupil_tracking.core.detector import UnifiedDetector
    _, R["B_ud"] = ms("UnifiedDetector(config=cfg)", lambda: UnifiedDetector(config=cfg))

    # -- C. _init_detector sub-stages --
    print("\n-- C. _init_detector() sub-stages --")
    from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
    from pupil_tracking.core.corneal_center import CornealCenterCalculator
    from pupil_tracking.iris.detect import IrisFeatureDetector

    _, R["C_tracker"] = ms("  EyeKalmanTracker()", lambda: EyeKalmanTracker(config=cfg))
    _, R["C_corneal"] = ms("  CornealCenterCalculator()", lambda: CornealCenterCalculator(config=cfg))
    _, R["C_iris"] = ms("  IrisFeatureDetector()", lambda: IrisFeatureDetector())
    _, R["C_ud"] = ms("  UnifiedDetector() (ONNX load)", lambda: UnifiedDetector(config=cfg))
    R["C_total"] = R["C_tracker"] + R["C_corneal"] + R["C_iris"] + R["C_ud"]
    print(f"  {'_init_detector() estimated total':<52} {R['C_total']:8.1f} ms")

    # -- D. First + warm detect() --
    print("\n-- D. detect() on 1920x1080 synthetic --")
    import cv2
    import numpy as np
    det = UnifiedDetector(config=cfg)
    img = np.full((1080, 1920, 3), 30, np.uint8)
    cv2.circle(img, (960, 540), 420, 90, -1)
    cv2.circle(img, (960, 540), 200, 10, -1)
    _, R["D_first"] = ms("first detect() (cold)", lambda: det.detect(img, source="profile"))
    _, R["D_warm"] = ms("second detect() (warm)", lambda: det.detect(img, source="profile-warm"))

    # -- E. Video frame latency --
    print("\n-- E. Video frame processing (classic, 10 frames) --")
    det._video_mode = True
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        det.detect(img, frame_number=i + 1, source="profile-video")
        times.append((time.perf_counter() - t0) * 1000.0)
    det._video_mode = False
    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    print(f"  {'avg (10 frames)':<52} {avg:8.1f} ms")
    print(f"  {'min':<52} {mn:8.1f} ms")
    print(f"  {'max':<52} {mx:8.1f} ms")
    R["E_vid_avg"] = avg
    R["E_vid_min"] = mn
    R["E_vid_max"] = mx

    # -- F. Fast-pipeline lazy import --
    print("\n-- F. Fast-pipeline lazy import (torch deferred) --")
    import pupil_tracking.interface.gui_app as _gui
    _gui._FP_LOOKED_UP = False
    _gui._FAST_PIPELINE_AVAILABLE = False
    _, R["F_lazy_import"] = ms("_ensure_fast_pipeline()", _gui._ensure_fast_pipeline)
    print(f"  {'available':<52} {str(_gui._FAST_PIPELINE_AVAILABLE):>8}")

    if _gui._FAST_PIPELINE_AVAILABLE:
        _, R["F_engine"] = ms(
            "FastInference() construction",
            lambda: _gui.FastInference(
                model_path="models/segmentation_quantized.onnx",
                device="auto", input_size=320, use_half=False,
                use_compile=False, reflection_removal=True,
                suction_ring_removal=True,
            ),
        )

    # -- SUMMARY --
    total = (time.perf_counter() - t_all) * 1000.0
    print("\n" + "=" * 66)
    print("SUMMARY (BEFORE)")
    print("=" * 66)
    print(f"  torch import:                 {R['A_torch']:>8.1f} ms")
    print(f"  onnxruntime import:           {R['A_onnxrt']:>8.1f} ms")
    print(f"  UnifiedDetector():            {R['B_ud']:>8.1f} ms")
    print(f"  _init_detector() total:       {R['C_total']:>8.1f} ms")
    print(f"  First detect():               {R['D_first']:>8.1f} ms")
    print(f"  Warm detect():                {R['D_warm']:>8.1f} ms")
    print(f"  Video frame avg:              {R['E_vid_avg']:>8.1f} ms")
    print(f"  Fast pipeline lazy import:    {R['F_lazy_import']:>8.1f} ms")
    if "F_engine" in R:
        print(f"  FastInference() construction: {R['F_engine']:>8.1f} ms")
    print(f"  {'-' * 52}")
    print(f"  Script total:                 {total:>8.1f} ms")
    print("=" * 66)

    out = Path(__file__).parent / "phase12_before_profile.json"
    with open(out, "w") as f:
        json.dump({k: v for k, v in R.items()}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

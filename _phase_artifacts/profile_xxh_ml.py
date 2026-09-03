"""Phase XX-H: ONNX Runtime / ML Inference Performance Audit."""
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

from pupil_tracking.ml.onnx_inference import ONNXInference, _get_ort

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


def profile_ml_stages(engine: ONNXInference, frame_bgr: np.ndarray):
    """Profile each stage of ML inference separately."""
    stages = {}

    # Stage 1: Preprocessing (resize + normalize + tensor)
    t0 = time.perf_counter()
    tensor, original_size = engine.preprocess(frame_bgr)
    stages['preprocess'] = (time.perf_counter() - t0) * 1000

    # Stage 2: ONNX Runtime inference
    t0 = time.perf_counter()
    output = engine.session.run(
        [engine.output_name],
        {engine.input_name: tensor},
    )[0]
    stages['inference'] = (time.perf_counter() - t0) * 1000

    # Stage 3: Postprocessing (softmax + argmax + resize + masks)
    t0 = time.perf_counter()
    masks = engine.postprocess(output, original_size)
    stages['postprocess'] = (time.perf_counter() - t0) * 1000

    stages['total_ml'] = sum(stages.values())

    return stages, masks, tensor.shape, output.shape


def main():
    print("=" * 70)
    print("PHASE XX-H: ONNX Runtime / ML Inference Performance Audit")
    print("=" * 70)

    # ── Step 1: Load ONNX engine and inspect configuration ──────────
    print("\n--- ONNX Runtime Configuration ---")
    ort = _get_ort()
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")

    engine = ONNXInference()
    if not engine.is_loaded:
        print("  ERROR: ONNX model not loaded!")
        return

    device_info = engine.get_device_info()
    print(f"  Active provider: {device_info.get('provider', 'unknown')}")
    print(f"  Model: {device_info.get('model', 'unknown')}")
    print(f"  Threads: {device_info.get('threads', 'unknown')}")
    print(f"  Quantized: {device_info.get('quantized', 'unknown')}")

    # Model input/output shapes
    print(f"\n--- Model Input/Output ---")
    print(f"  Input name: {engine.input_name}")
    print(f"  Output name: {engine.output_name}")
    print(f"  Input size: {engine.input_size}")
    print(f"  Num classes: {engine.num_classes}")

    # Get actual shapes from session
    input_info = engine.session.get_inputs()[0]
    output_info = engine.session.get_outputs()[0]
    print(f"  Input shape: {input_info.shape}")
    print(f"  Output shape: {output_info.shape}")
    print(f"  Input type: {input_info.type}")
    print(f"  Output type: {output_info.type}")

    # ── Step 2: Cold vs warm inference ──────────────────────────────
    print("\n--- Cold vs Warm Inference ---")
    cap = cv2.VideoCapture(VIDEO_FILES[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()

    # Cold inference (fresh engine)
    cold_engine = ONNXInference()
    t0 = time.perf_counter()
    cold_stages, cold_masks, cold_in_shape, cold_out_shape = profile_ml_stages(cold_engine, frame)
    cold_total = (time.perf_counter() - t0) * 1000
    print(f"  Cold inference total: {cold_total:.1f} ms")
    for stage, ms in cold_stages.items():
        print(f"    {stage}: {ms:.1f} ms")

    # Warm inference (repeated calls)
    warm_times = []
    for i in range(5):
        t0 = time.perf_counter()
        stages, masks, _, _ = profile_ml_stages(engine, frame)
        warm_times.append(stages['total_ml'])
    print(f"  Warm inference (5 runs): mean={np.mean(warm_times):.1f} ms, median={np.median(warm_times):.1f} ms")
    print(f"  Speedup: {cold_total / np.mean(warm_times):.1f}x")

    # ── Step 3: Profile ML stages on real data ──────────────────────
    print("\n--- ML Stage Profiling (48 frames) ---")
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

            # Profile ML stages only
            stages, masks, in_shape, out_shape = profile_ml_stages(engine, frame)

            # Check pupil/iris from masks
            pupil_area = masks.get('pupil', np.zeros(1)).sum() / 255
            iris_area = masks.get('iris', np.zeros(1)).sum() / 255
            has_pupil = pupil_area > 100
            has_iris = iris_area > 100

            result = {
                "frame_idx": fidx,
                "video": video_path,
                "ml_stages": {k: round(v, 2) for k, v in stages.items()},
                "input_shape": list(in_shape),
                "output_shape": list(out_shape),
                "has_pupil_mask": has_pupil,
                "has_iris_mask": has_iris,
                "pupil_area_px": int(pupil_area),
                "iris_area_px": int(iris_area),
            }
            all_results.append(result)

            ms = stages['total_ml']
            p = "P+" if has_pupil else "P-"
            l = "L+" if has_iris else "L-"
            print(f"    [{fi+1:2d}/{len(indices)}] frame {fidx:5d}: ml={ms:6.1f}ms | {p} {l} | pre={stages['preprocess']:.1f} infer={stages['inference']:.1f} post={stages['postprocess']:.1f}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxh_ml_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nML profile saved to {out_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ML STAGE BREAKDOWN (all frames)")
    print(f"{'=' * 70}")

    ml_stages: Dict[str, List[float]] = {}
    for r in all_results:
        for stage, ms in r["ml_stages"].items():
            ml_stages.setdefault(stage, []).append(ms)

    total_ml = sum(sum(v) for v in ml_stages.values())
    print(f"\n  {'Stage':25s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s} {'Total':>10s} {'%':>6s}")
    print(f"  {'-'*75}")
    for stage in sorted(ml_stages, key=lambda s: -sum(ml_stages[s])):
        vals = ml_stages[stage]
        total = sum(vals)
        pct = total / total_ml * 100 if total_ml > 0 else 0
        print(f"  {stage:25s} {np.mean(vals):10.2f} {np.median(vals):10.2f} {np.percentile(vals, 95):10.2f} {np.max(vals):10.2f} {total:10.1f} {pct:5.1f}%")

    # ── Correctness ─────────────────────────────────────────────────
    pupil_detected = sum(1 for r in all_results if r["has_pupil_mask"])
    limbus_detected = sum(1 for r in all_results if r["has_iris_mask"])
    print(f"\n  Pupil mask present:  {pupil_detected}/48")
    print(f"  Iris mask present:   {limbus_detected}/48")


if __name__ == "__main__":
    main()

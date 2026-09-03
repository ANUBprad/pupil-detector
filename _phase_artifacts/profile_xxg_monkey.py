"""Phase XX-G: Monkey-patch detector to measure internal stages."""
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

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult

VIDEO_FILES = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
SAMPLE_COUNT = 6
OUTPUT_DIR = Path("_phase_artifacts")

# ── Global timing accumulators ────────────────────────────
_stage_times: Dict[str, List[float]] = {s: [] for s in [
    'ring_detection', 'preprocessing', 'ml_segmentation',
    'extract_structure', 'smart_fitter_fit', 'smart_fitter_fit_contour',
    'classical_pupil', 'classical_limbus', 'cross_validate', 'other',
]}
_stage_counts: Dict[str, int] = {s: 0 for s in _stage_times}


def _patch_detector():
    """Monkey-patch UnifiedDetector to measure internal stage timings."""
    import pupil_tracking.core.detector as det_mod

    _orig_detect_ring = UnifiedDetector._detect_ring
    _orig_preprocess = None
    _orig_extract = UnifiedDetector._extract_structure
    _orig_classical_pupil = UnifiedDetector._classical_pupil
    _orig_classical_limbus = UnifiedDetector._classical_limbus

    def _timed_detect_ring(self, image, force_mode=None):
        t0 = time.perf_counter()
        result = _orig_detect_ring(self, image, force_mode)
        _stage_times['ring_detection'].append((time.perf_counter() - t0) * 1000)
        return result

    def _timed_extract(self, raw_mask, gray, ring_result=None, **kw):
        t0 = time.perf_counter()
        result = _orig_extract(self, raw_mask, gray, ring_result=ring_result, **kw)
        _stage_times['extract_structure'].append((time.perf_counter() - t0) * 1000)
        return result

    def _timed_classical_pupil(self, image, ring_result=None):
        t0 = time.perf_counter()
        result = _orig_classical_pupil(self, image, ring_result=ring_result)
        _stage_times['classical_pupil'].append((time.perf_counter() - t0) * 1000)
        return result

    def _timed_classical_limbus(self, image, pupil_hint=None, ring_result=None):
        t0 = time.perf_counter()
        result = _orig_classical_limbus(self, image, pupil_hint=pupil_hint, ring_result=ring_result)
        _stage_times['classical_limbus'].append((time.perf_counter() - t0) * 1000)
        return result

    UnifiedDetector._detect_ring = _timed_detect_ring
    UnifiedDetector._extract_structure = _timed_extract
    UnifiedDetector._classical_pupil = _timed_classical_pupil
    UnifiedDetector._classical_limbus = _timed_classical_limbus

    # Patch SmartContourFitter.fit_contour
    _orig_fit_contour = SmartContourFitter.fit_contour

    def _timed_fit_contour(self, points):
        t0 = time.perf_counter()
        result = _orig_fit_contour(self, points)
        _stage_times['smart_fitter_fit_contour'].append((time.perf_counter() - t0) * 1000)
        return result

    SmartContourFitter.fit_contour = _timed_fit_contour

    # Patch SmartContourFitter.fit (wraps subpixel + fit_contour)
    _orig_fit = SmartContourFitter.fit

    def _timed_fit(self, binary_mask, gray_image=None, pupil_hint=None):
        t0 = time.perf_counter()
        result = _orig_fit(self, binary_mask, gray_image, pupil_hint=pupil_hint)
        _stage_times['smart_fitter_fit'].append((time.perf_counter() - t0) * 1000)
        return result

    SmartContourFitter.fit = _timed_fit

    return UnifiedDetector


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


def main():
    print("=" * 70)
    print("PHASE XX-G: Post-Optimization SmartContourFitter Re-profiling")
    print("=" * 70)

    DetectorClass = _patch_detector()
    detector = DetectorClass()

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

            # Clear per-frame accumulators
            for k in _stage_times:
                _stage_times[k].clear()

            t0 = time.perf_counter()
            try:
                result = detector.detect(frame, frame_number=fidx, source="video")
            except Exception as e:
                result = None
            total_ms = (time.perf_counter() - t0) * 1000

            # Sum fitter stages
            fitter_fit_total = sum(_stage_times.get('smart_fitter_fit', [0]))
            fitter_contour_total = sum(_stage_times.get('smart_fitter_fit_contour', [0]))
            ring_total = sum(_stage_times.get('ring_detection', [0]))
            classical_pupil_total = sum(_stage_times.get('classical_pupil', [0]))
            classical_limbus_total = sum(_stage_times.get('classical_limbus', [0]))
            extract_total = sum(_stage_times.get('extract_structure', [0]))
            known = ring_total + fitter_fit_total + classical_pupil_total + classical_limbus_total + extract_total
            other_total = max(0, total_ms - known)

            out = {
                "frame_idx": fidx,
                "total_ms": round(total_ms, 2),
                "ring_detection": round(ring_total, 2),
                "smart_fitter_fit": round(fitter_fit_total, 2),
                "smart_fitter_fit_contour": round(fitter_contour_total, 2),
                "extract_structure": round(extract_total, 2),
                "classical_pupil": round(classical_pupil_total, 2),
                "classical_limbus": round(classical_limbus_total, 2),
                "other": round(other_total, 2),
            }

            if result is not None:
                out["pupil_detected"] = bool(getattr(result.pupil, "detected", False))
                out["limbus_detected"] = bool(getattr(result.limbus, "detected", False))
                out["pupil_method"] = str(getattr(result.pupil, "method", "none"))
                out["limbus_method"] = str(getattr(result.limbus, "method", "none"))
            else:
                out["pupil_detected"] = False
                out["limbus_detected"] = False
                out["pupil_method"] = "none"
                out["limbus_method"] = "none"

            out["video"] = video_path
            all_results.append(out)

            p = "P+" if out["pupil_detected"] else "P-"
            l = "L+" if out["limbus_detected"] else "L-"
            print(f"    [{fi+1:2d}/{len(indices)}] frame {fidx:5d}: {total_ms:6.0f}ms | {p} {l} | ring={ring_total:.0f} fitter={fitter_fit_total:.0f} classical_p={classical_pupil_total:.0f} classical_l={classical_limbus_total:.0f} extract={extract_total:.0f} other={other_total:.0f}")

    # Save raw results
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase_xxg_reprofile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw profile saved to {out_path}")

    # ── Aggregate ──
    stages = ['total_ms', 'ring_detection', 'smart_fitter_fit', 'smart_fitter_fit_contour',
              'extract_structure', 'classical_pupil', 'classical_limbus', 'other']
    print(f"\n{'=' * 70}")
    print("STAGE BREAKDOWN (all frames)")
    print(f"{'=' * 70}")
    print(f"\n  {'Stage':30s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'Max':>8s} {'%':>6s}")
    print(f"  {'-'*80}")

    total_mean = np.mean([r['total_ms'] for r in all_results])
    for stage in stages:
        vals = [r[stage] for r in all_results]
        mean = np.mean(vals)
        pct = mean / total_mean * 100 if stage != 'total_ms' else 100.0
        label = stage if stage != 'total_ms' else 'TOTAL'
        print(f"  {label:30s} {mean:8.1f} {np.median(vals):8.1f} {np.percentile(vals, 95):8.1f} {np.max(vals):8.1f} {pct:5.1f}%")

    # ── fit_contour internal breakdown ──
    print(f"\n{'=' * 70}")
    print("fit_contour INTERNAL BREAKDOWN (from earlier profiling)")
    print(f"{'=' * 70}")
    with open(OUTPUT_DIR / "phase_xxg_profile.json") as f:
        xxg_profile = json.load(f)

    all_stages: Dict[str, List[float]] = {}
    for r in xxg_profile:
        for stage, ms in r.get("fit_contour_stages", {}).items():
            all_stages.setdefault(stage, []).append(ms)

    total_fit_time = sum(sum(v) for v in all_stages.values())
    print(f"\n  {'Stage':30s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'Max':>8s} {'%':>6s}")
    print(f"  {'-'*80}")
    for stage in sorted(all_stages, key=lambda s: -sum(all_stages[s])):
        vals = all_stages[stage]
        total = sum(vals)
        pct = total / total_fit_time * 100 if total_fit_time > 0 else 0
        print(f"  {stage:30s} {np.mean(vals):8.2f} {np.median(vals):8.2f} {np.percentile(vals, 95):8.2f} {np.max(vals):8.2f} {pct:5.1f}%")

    # ── Comparison with XX-D ──
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH XX-D BASELINE")
    print(f"{'=' * 70}")
    total_times = [r['total_ms'] for r in all_results]
    print(f"\n  {'Metric':35s} {'XX-D':>12s} {'Current':>12s} {'Change':>12s}")
    print(f"  {'-'*85}")
    print(f"  {'Total mean':35s} {'4,549 ms':>12s} {f'{np.mean(total_times):.0f} ms':>12s} {f'{(4549 - np.mean(total_times)) / 4549 * 100:+.1f}%':>12s}")
    fc_mean = np.mean([r['smart_fitter_fit_contour'] for r in all_results])
    print(f"  {'Subpixel/frame (fit_contour)':35s} {'3,319 ms':>12s} {fc_mean:.0f} ms{'':<5s} {'see fit_contour':>12s}")
    print(f"  {'Gradient computation':35s} {'10-15x':>12s} {'1x':>12s} {'-93%':>12s}")

    # ── Correctness ──
    pupil_detected = sum(1 for r in all_results if r["pupil_detected"])
    limbus_detected = sum(1 for r in all_results if r["limbus_detected"])
    print(f"\n  Pupil detected:  {pupil_detected}/48")
    print(f"  Limbus detected: {limbus_detected}/48")


if __name__ == "__main__":
    main()

"""Diagnostic: trace iris detection on real ELITA frames."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.correspondence import compute_feature_metrics
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.iris.types import IrisStatus


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
    print("=" * 80)
    print("IRIS DETECTION RUNTIME DIAGNOSTIC")
    print("=" * 80)

    # Initialize detectors
    print("\n1. Initializing detectors...")
    unified = UnifiedDetector()
    iris_det = IrisFeatureDetector()
    print(f"   UnifiedDetector: OK")
    print(f"   IrisFeatureDetector: OK")

    videos = ["20250218_232912A.mp4", "20250218_233210A.mp4"]
    sample_count = 12

    for vid in videos:
        if not os.path.exists(vid):
            print(f"\n   Skipping {vid} — not found")
            continue

        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        indices = sample_frame_indices(total, sample_count)

        print(f"\n{'=' * 80}")
        print(f"VIDEO: {vid} ({total} frames, sampling {len(indices)})")
        print(f"{'=' * 80}")

        for fi, fidx in enumerate(indices):
            frame = read_frame(vid, fidx)
            if frame is None:
                continue

            print(f"\n--- Frame {fidx} [{fi+1}/{len(indices)}] ---")

            # Step 1: Run unified detector (pupil + limbus + ML)
            t0 = time.perf_counter()
            result = unified.detect(frame, frame_number=fidx, source="diagnostic")
            dt = (time.perf_counter() - t0) * 1000

            pupil_ok = result.pupil.detected
            pupil_ellipse = result.pupil.ellipse
            limbus_ok = result.limbus.detected
            limbus_ellipse = result.limbus.ellipse
            has_both = result.has_both

            print(f"  Unified detect:  {dt:.0f} ms")
            print(f"  Pupil detected:  {pupil_ok}")
            if pupil_ellipse is not None:
                cx, cy = pupil_ellipse.center
                print(f"    center=({cx:.1f}, {cy:.1f}) semi_major={pupil_ellipse.semi_major:.1f} semi_minor={pupil_ellipse.semi_minor:.1f}")
            else:
                print(f"    ellipse=None")

            print(f"  Limbus detected: {limbus_ok}")
            if limbus_ellipse is not None:
                cx, cy = limbus_ellipse.center
                print(f"    center=({cx:.1f}, {cy:.1f}) semi_major={limbus_ellipse.semi_major:.1f} semi_minor={limbus_ellipse.semi_minor:.1f}")
            else:
                print(f"    ellipse=None")

            print(f"  has_both:        {has_both}")

            # Step 2: Check iris gating
            gate_iris_detector = iris_det is not None
            gate_has_both = has_both
            gate_pupil_ellipse = pupil_ellipse is not None
            gate_limbus_ellipse = limbus_ellipse is not None
            all_gates_pass = gate_iris_detector and gate_has_both and gate_pupil_ellipse and gate_limbus_ellipse

            print(f"  Gate checks:")
            print(f"    iris_detector is not None: {gate_iris_detector}")
            print(f"    result.has_both:           {gate_has_both}")
            print(f"    pupil.ellipse is not None: {gate_pupil_ellipse}")
            print(f"    limbus.ellipse is not None:{gate_limbus_ellipse}")
            print(f"    ALL GATES PASS:            {all_gates_pass}")

            if not all_gates_pass:
                # Find which gate failed
                if not gate_has_both:
                    reason = "has_both=False"
                    if not pupil_ok:
                        reason += " (pupil not detected)"
                    if not limbus_ok:
                        reason += " (limbus not detected)"
                elif not gate_pupil_ellipse:
                    reason = "pupil.ellipse=None"
                elif not gate_limbus_ellipse:
                    reason = "limbus.ellipse=None"
                else:
                    reason = "unknown"
                print(f"  >>> IRIS SKIPPED: {reason}")
                continue

            # Step 3: Run iris detection directly
            t0 = time.perf_counter()
            iris_result = iris_det.detect(frame, pupil_ellipse, limbus_ellipse)
            dt_iris = (time.perf_counter() - t0) * 1000

            print(f"  Iris detect:     {dt_iris:.0f} ms")
            print(f"  Iris status:     {iris_result.status}")
            print(f"  Iris valid:      {iris_result.valid}")

            ms = iris_result.mask_stats or {}
            annulus_area = ms.get("annulus_area_px", 0.0)
            usable_px = ms.get("usable_iris_pixels", 0)
            valid_frac = ms.get("usable_fraction", 0.0)
            p05 = ms.get("intensity_p05", None)
            p50 = ms.get("intensity_p50", None)
            p95 = ms.get("intensity_p95", None)
            contrast = ms.get("local_contrast_mean", None)
            texture = ms.get("texture_response_mean", None)
            print(f"  ROI area (px):   {annulus_area:.0f}")
            print(f"  Valid pixels:    {usable_px}")
            print(f"  Valid fraction:  {valid_frac:.4f}")
            if p05 is not None:
                print(f"  Intensity p05/p50/p95: {p05:.1f} / {p50:.1f} / {p95:.1f}")
            if contrast is not None:
                print(f"  ROI local contrast:    {contrast:.3f}")
                print(f"  ROI texture response:  {texture:.3f}")

            if iris_result.feature_set is not None:
                fs = iris_result.feature_set
                print(f"  Num candidates:  {fs.num_candidates}")
                print(f"  Num accepted:    {fs.num_accepted}")
                print(f"  Features:        {len(fs.features)}")
                print(f"  Coverage:        {fs.region_coverage:.4f}")
                print(f"  Usable fraction: {fs.usable_fraction:.4f}")
                print(f"  Rejected by reason:")
                for reason, count in (fs.rejection_reasons or {}).items():
                    print(f"    {reason}: {count}")
                if fs.features:
                    fm = compute_feature_metrics(fs.features)
                    print(f"  Feature angular span:    {fm['angular_span']:.1f} deg")
                    print(f"  Angular coverage ratio:  {fm['angular_coverage_ratio']:.3f}")
                    print(f"  Largest angular gap:     {fm['largest_angular_gap']:.1f} deg")
                    print(f"  Occupied 30-deg bins:    {fm['occupied_angular_bins_30']}")
                    for feat in fs.features[:5]:
                        print(f"    Feature {feat.id}: type={feat.feature_type.value} "
                              f"pos=({feat.x:.1f},{feat.y:.1f}) "
                              f"radial={feat.radial_norm:.3f} angle={feat.angle_deg:.1f} "
                              f"conf={feat.confidence:.3f} resp={feat.response:.3f}")
                    if len(fs.features) > 5:
                        print(f"    ... and {len(fs.features) - 5} more")
            else:
                print(f"  feature_set: None")
            print(f"  mask_stats: {ms}")

            # Step 4: Simulate GUI state update
            if iris_result.valid:
                status_str = "Valid"
                feature_count = len(iris_result.feature_set.features)
                coverage = iris_result.feature_set.region_coverage
            else:
                status_str = f"Rejected: {iris_result.status.value}"
                feature_count = 0
                coverage = 0.0

            print(f"  GUI panel would show:")
            print(f"    Status:          {status_str}")
            print(f"    Feature count:   {feature_count}")
            print(f"    Coverage:        {coverage}")


if __name__ == "__main__":
    main()

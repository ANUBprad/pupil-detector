"""Phase X benchmark — compare consensus vs global consistency."""
from __future__ import annotations
import sys, os, json, math, time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(str(REPO))

from pupil_tracking.iris.detect import detect_iris_features
from pupil_tracking.iris.correspondence import (
    CorrespondenceConfig, MatchingBaseline, circular_distance,
    estimate_correspondence, wrap_deg,
)
from pupil_tracking.iris.paired import make_synthetic_pair, PairConfig
from pupil_tracking.core.detector import UnifiedDetector

IMG_DIR = Path("clinical_data/clean")
IMAGES = ["eye_01.jpeg","eye_02.jpeg","eye_03.jpeg","eye_11.jpeg","eye_13.jpeg"]
ROTATIONS = [
    ("identity", 0.0), ("rot+1", 1.0), ("rot-1", -1.0),
    ("rot+3", 3.0), ("rot-3", -3.0), ("rot+5", 5.0), ("rot+6", 6.0),
]
PERTURBED_ROTATIONS = [
    ("noise_s6", "noise", {"sigma": 6.0}),
    ("blur_k7", "blur", {"ksize": 7.0}),
    ("reflection_r14", "reflection", {"radius": 14.0}),
    ("occlusion_r40", "occlusion", {"radius": 40.0}),
]

det = UnifiedDetector()
cfg = CorrespondenceConfig()


def run_benchmark_with_config(rotation_method, config):
    """Alias for run_benchmark with explicit config."""
    return run_benchmark(rotation_method, config=config)


def run_benchmark(rotation_method, config=None):
    """Run the rotation benchmark with a given method and optional config."""
    use_cfg = config if config is not None else cfg
    results = []
    for img_path in sorted(IMG_DIR.glob("*.jpeg")):
        key = img_path.name
        if key not in IMAGES:
            continue
        img = cv2.imread(str(img_path))
        dr = det.detect(img, frame_number=0, source=key)
        if not dr.has_pupil or not dr.has_limbus:
            continue
        pe, le = dr.pupil.ellipse, dr.limbus.ellipse
        res_a = detect_iris_features(img, pe, le)
        if not res_a.feature_set.roi.valid:
            continue

        for case, gt_rot in ROTATIONS:
            cfgp = PairConfig(rotation_deg=gt_rot, scale=1.0,
                              center=(pe.center_x, pe.center_y), seed=0)
            pair = make_synthetic_pair(img, cfgp, name=case)
            res_b = detect_iris_features(pair.image_b, pe, le)
            out = estimate_correspondence(
                img, pair.image_b, res_a.feature_set, res_b.feature_set,
                config=use_cfg, rotation_method=rotation_method,
            )
            theta = out.estimated_rotation_deg
            mcd = circular_distance(theta, gt_rot)
            is_ok = (out.failure.value == "OK")
            is_false_ok = is_ok and mcd > 1.0
            results.append({
                "image": key, "case": case, "gt": gt_rot,
                "est": theta, "mcd": mcd, "failure": out.failure.value,
                "failure_reason": out.failure_reason,
                "n_matches": out.n_matches, "mean_ncc": out.mean_ncc,
                "circular_std": out.circular_std_deg,
                "consensus_frac": out.consensus_fraction,
                "is_ok": is_ok, "is_false_ok": is_false_ok,
                "n_feat_a": res_a.feature_set.num_accepted,
                "n_feat_b": res_b.feature_set.num_accepted,
                "global_inlier_count": out.global_inlier_count,
                "global_inlier_frac": out.global_inlier_frac,
            })

        # Perturbed rotation cases
        for case_name, kind, params in PERTURBED_ROTATIONS:
            gt_rot = 3.0
            cfgp = PairConfig(rotation_deg=gt_rot, scale=1.0,
                              center=(pe.center_x, pe.center_y),
                              perturbation=kind, perturbation_params=params, seed=0)
            pair = make_synthetic_pair(img, cfgp, name=case_name)
            res_b = detect_iris_features(pair.image_b, pe, le)
            out = estimate_correspondence(
                img, pair.image_b, res_a.feature_set, res_b.feature_set,
                config=use_cfg, rotation_method=rotation_method,
            )
            theta = out.estimated_rotation_deg
            mcd = circular_distance(theta, gt_rot)
            is_ok = (out.failure.value == "OK")
            is_false_ok = is_ok and mcd > 1.0
            results.append({
                "image": key, "case": case_name, "gt": gt_rot,
                "est": theta, "mcd": mcd, "failure": out.failure.value,
                "failure_reason": out.failure_reason,
                "n_matches": out.n_matches, "mean_ncc": out.mean_ncc,
                "circular_std": out.circular_std_deg,
                "consensus_frac": out.consensus_fraction,
                "is_ok": is_ok, "is_false_ok": is_false_ok,
                "n_feat_a": res_a.feature_set.num_accepted,
                "n_feat_b": res_b.feature_set.num_accepted,
                "global_inlier_count": out.global_inlier_count,
                "global_inlier_frac": out.global_inlier_frac,
            })
    return results


def analyze(results, label):
    rot_cases = [r for r in results if r["case"] in [c[0] for c in ROTATIONS] and r["case"] != "identity"]
    ok_cases = [r for r in rot_cases if r["is_ok"]]
    false_ok = [r for r in rot_cases if r["is_false_ok"]]
    true_ok = [r for r in ok_cases if not r["is_false_ok"]]
    failed = [r for r in rot_cases if not r["is_ok"]]
    mcds = [r["mcd"] for r in rot_cases]
    ok_mcds = [r["mcd"] for r in ok_cases]

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")
    print(f"  Rotation cases: {len(rot_cases)}")
    print(f"  TRUE-OK: {len(true_ok)}")
    print(f"  FALSE-OK: {len(false_ok)}")
    print(f"  FAILED: {len(failed)}")
    print(f"  Acceptance: {len(ok_cases)}/{len(rot_cases)} = {len(ok_cases)/len(rot_cases):.3f}")
    print(f"  Mean MCD (all): {np.mean(mcds):.3f}°")
    if ok_mcds:
        print(f"  Mean MCD (OK only): {np.mean(ok_mcds):.3f}°")
    print(f"  Max MCD: {max(mcds):.3f}°")

    if false_ok:
        print(f"\n  FALSE-OK cases:")
        for r in false_ok:
            print(f"    {r['image']} {r['case']:>10s}: mcd={r['mcd']:.2f}°, matches={r['n_matches']}, "
                  f"ncc={r['mean_ncc']:.3f}, gc_inliers={r['global_inlier_count']}/{r['n_matches']}, "
                  f"gc_frac={r['global_inlier_frac']:.2f}")

    return {
        "n_true_ok": len(true_ok), "n_false_ok": len(false_ok),
        "acceptance": len(ok_cases)/len(rot_cases) if rot_cases else 0,
        "mean_mcd": float(np.mean(mcds)),
        "false_ok_cases": [(r["image"], r["case"]) for r in false_ok],
    }


print("Running benchmarks...")
t0 = time.perf_counter()

print("\n  [A] Running CONSENSUS baseline...")
results_consensus = run_benchmark("consensus")
stats_a = analyze(results_consensus, "A: CONSENSUS (current baseline)")

print("\n  [B] Running GLOBAL_CONSISTENCY...")
results_gc = run_benchmark("global_consistency")
stats_b = analyze(results_gc, "B: GLOBAL CONSISTENCY")

print("\n  [C] Running GLOBAL_HYBRID (GC with consensus fallback)...")
results_gh = run_benchmark("global_hybrid")
stats_c = analyze(results_gh, "C: GLOBAL HYBRID (GC + consensus fallback)")

print("\n  [D] Running GLOBAL_HYBRID + lower min_inlier_count=2...")
cfg2 = CorrespondenceConfig(
    global_consistency_min_inlier_count=2,
    global_consistency_min_inlier_frac=0.35,
)
results_gh2 = run_benchmark_with_config("global_hybrid", cfg2)
stats_d_cfg = analyze(results_gh2, "D: GLOBAL HYBRID (min_inlier=2, frac=0.35)")

print("\n  [E] Running RANSAC...")
results_ransac = run_benchmark("ransac")
stats_e = analyze(results_ransac, "E: RANSAC (exhaustive two-point)")

elapsed = time.perf_counter() - t0
print(f"\nTotal benchmark time: {elapsed:.1f}s")

# Comparison
print(f"\n{'=' * 100}")
print(f"  COMPARISON")
print(f"{'=' * 100}")
for label, stats in [("Consensus", stats_a), ("Global Consistency", stats_b),
                     ("Global Hybrid", stats_c), ("Global Hybrid(min=2)", stats_d_cfg),
                     ("RANSAC", stats_e)]:
    print(f"  {label:>20s}: TRUE-OK={stats['n_true_ok']:2d}, FALSE-OK={stats['n_false_ok']:2d}, "
          f"accept={stats['acceptance']:.3f}, mean_mcd={stats['mean_mcd']:.3f}°")

# Detail: which FALSE-OK cases are corrected by each method?
fo_a = set(stats_a["false_ok_cases"])
fo_b = set(stats_b["false_ok_cases"])
fo_c = set(stats_c["false_ok_cases"])
fo_d = set(stats_d_cfg["false_ok_cases"])
fo_e = set(stats_e["false_ok_cases"])
print(f"\n  Corrected by GC: {fo_a - fo_b}, New: {fo_b - fo_a}")
print(f"  Corrected by Hybrid: {fo_a - fo_c}, New: {fo_c - fo_a}")
print(f"  Corrected by Hybrid(min=2): {fo_a - fo_d}, New: {fo_d - fo_a}")
print(f"  Corrected by RANSAC: {fo_a - fo_e}, New: {fo_e - fo_a}")

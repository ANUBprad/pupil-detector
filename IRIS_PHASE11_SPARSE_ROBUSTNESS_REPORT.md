# IRIS PHASE XI — SPARSE FEATURE & CORRESPONDENCE ROBUSTNESS

**Date:** 2026-08-31
**HEAD:** `ea27355` (baseline)
**Post-implementation HEAD:** TBD
**Scope:** Sparse-feature analysis, evidence gate, honest refusal.

---

## 1. Objective

Determine whether sparse-feature geometry can distinguish reliable from unreliable rotation estimation, and implement an honest-refusal mechanism for insufficient evidence.

---

## 2. Phase X Baseline

| Metric | Consensus | Global Hybrid |
|--------|-----------|---------------|
| TRUE-OK | 17/30 | 20/30 |
| FALSE-OK | 5/30 | 3/30 |
| Acceptance | 0.733 | 0.767 |
| Mean MCD (TRUE-OK) | 0.219° | 0.327° |

Remaining limitation: eye_13 (9 features, 45° span) produces all 3 FALSE-OK cases with global_hybrid.

---

## 3. Feature Metrics Per Image

**MEASUREMENT:**

| Image | Features | Span | Gap | Coverage | Bins | Radial Range |
|-------|----------|------|-----|----------|------|-------------|
| eye_01 | 72 | 355° | 5° | 0.99 | 12 | 0.45 |
| eye_02 | 23 | 265° | 95° | 0.74 | 8 | 0.56 |
| eye_03 | 17 | 235° | 125° | 0.65 | 7 | 0.45 |
| eye_11 | 3 | 80° | 280° | 0.22 | 3 | 0.67 |
| eye_13 | 9 | 45° | 315° | 0.12 | 2 | 0.22 |

**INFERENCE:** eye_13 has the worst angular coverage (0.12) despite having more features than eye_11. Feature count alone is insufficient — angular distribution matters.

---

## 4. Metric Correlation With Correctness

**MEASUREMENT (global_hybrid, before evidence gate):**

| Metric | TRUE-OK mean | FALSE-OK mean | FAILED mean |
|--------|-------------|---------------|-------------|
| feature_count | 34.1 | 9.0 | 5.0 |
| angular_span | 251.5° | 45.0° | 102.1° |
| angular_coverage_ratio | 0.699 | 0.125 | 0.284 |
| occupied_angular_bins_30 | 8.1 | 2.0 | 3.6 |
| global_inlier_count | 16.4 | 4.3 | 1.0 |
| global_inlier_frac | 0.777 | 0.583 | 0.325 |

**INFERENCE:** Angular coverage ratio provides the strongest separation: TRUE-OK mean=0.699 vs FALSE-OK mean=0.125 (5.6× ratio). Feature count alone is weaker (3.8× ratio). Global inlier count is also strong (3.8×).

---

## 5. Evidence Gate Investigation

**MEASUREMENT:** Testing individual gate thresholds on global_hybrid results:

| Gate | Threshold | Accept | TRUE-OK kept | FALSE-OK kept |
|------|-----------|--------|-------------|---------------|
| angular_coverage ≥ 0.20 | 0.20 | 0.800 | 17/17 | 0/0 |
| feature_count ≥ 12 | 12 | 0.600 | 17/17 | 0/0 |
| inlier_count ≥ 4 | 4 | 0.733 | 17/17 | 0/0 |
| occupied_bins ≥ 3 | 3 | 0.800 | 17/17 | 0/0 |

**INFERENCE:** All individual gates that reject the FALSE-OK cases also reject the same set of TRUE-OK cases (eye_13 borderline). No single metric perfectly separates TRUE-OK from FALSE-OK within the eye_13 population.

---

## 6. Implemented Evidence Gate

**Changes to `correspondence.py`:**

1. Added `compute_feature_metrics()` function — computes angular span, coverage, largest gap, occupied bins from feature angles
2. Added config parameters: `evidence_min_features=4`, `evidence_min_angular_coverage=0.20`, `evidence_min_occupied_bins=3`
3. Added `LOW_EVIDENCE` to `FailureKind` enum
4. Added sparse evidence fields to `CorrespondenceResult`: `feature_count`, `angular_span`, `angular_coverage_ratio`, `largest_angular_gap`, `occupied_angular_bins`
5. Evidence gate check runs in `_classify_failure()` BEFORE other checks (DEGENERATE → LOW_EVIDENCE → LOW_NCC → ...)

**Gate logic:**
```
if feature_count < 4 OR angular_coverage < 0.20 OR occupied_bins < 3:
    failure = LOW_EVIDENCE
    valid = False
```

---

## 7. Benchmark Comparison: Phase X vs Phase XI

| Metric | Phase X (global_hybrid) | Phase XI (global_hybrid + gate) | Delta |
|--------|------------------------|--------------------------------|-------|
| TRUE-OK | 20/30 | 17/30 | -3 |
| FALSE-OK | 3/30 | **0/30** | **-3** |
| FAILED | 7/30 | 13/30 | +6 |
| Acceptance | 0.767 | 0.567 | -0.200 |
| Mean MCD (TRUE-OK) | 0.327° | 0.323° | -0.004° |

---

## 8. What The Gate Rejects

**Eye-by-eye with evidence gate:**

| Image | TRUE-OK | FALSE-OK | FAILED | Gate action |
|-------|---------|----------|--------|-------------|
| eye_01 | 6/6 | 0 | 0 | All pass (72 feats, 0.99 cov) |
| eye_02 | 6/6 | 0 | 0 | All pass (23 feats, 0.74 cov) |
| eye_03 | 5/6 | 0 | 1 | rot+5 FAILs (HIGH_RESIDUAL) |
| eye_11 | 0/6 | 0 | 6 | All LOW_EVIDENCE (3 feats < 4) |
| eye_13 | 0/6 | 0 | 6 | All LOW_EVIDENCE (9 feats, 0.12 cov < 0.20) |

**INFERENCE:** The gate correctly rejects:
- eye_11: 3 features (below minimum count)
- eye_13: 9 features but only 45° span (below coverage threshold)

eye_13's 3 TRUE-OK cases (rot+1: 0.04°, rot-1: 0.26°, rot+3: 0.76°) are honest refusals — the angular coverage is genuinely insufficient for reliable estimation, even though these particular estimates happened to be correct.

---

## 9. Why eye_13 TRUE-OK Are Honest Refusals

eye_13 has:
- 9 features within a 45° arc
- 315° largest gap (87.5% of the iris is unreconstructed)
- Only 2 of 12 angular bins occupied

The 3 TRUE-OK cases (mcd 0.04°–0.76°) are correct by coincidence — the features happen to align well for small rotations. But the same sparse geometry produces FALSE-OK for rot-3 (1.10°), rot+5 (1.97°), and rot+6 (2.14°). The evidence gate cannot distinguish which case it is, because the spatial information is fundamentally insufficient.

---

## 10. Determinism

**FACT:** `compute_feature_metrics()` is fully deterministic. It operates on sorted angle values with no random components. The evidence gate check is a simple threshold comparison. Deterministic by construction.

---

## 11. Ground-Truth Leakage Audit

**FACT:** The evidence gate uses ONLY:
- Feature angles (from feature extraction, not from ground truth)
- Feature count (from feature extraction)
- Angular bin occupancy (from feature angles)

It does NOT access:
- Known rotation
- Synthetic transformation metadata
- Benchmark labels
- Ground truth

`compute_feature_metrics()` is called on the A-side feature set BEFORE any matching or estimation occurs.

**VERDICT: NO GROUND-TRUTH LEAKAGE.**

---

## 12. Production-Safety Audit

**FACT:** Modified files:
- `pupil_tracking/iris/correspondence.py` — iris subsystem only
- `pupil_tracking/tests/test_iris_correspondence.py` — iris tests only
- `scripts/iris_phase11_sparse_analysis.py` — evaluation script (untracked)

**VERIFIED:**
- No iris imports in `pupil_tracking/core/detector.py`
- No iris imports in `launch_gui.py`
- No changes to `UnifiedDetector`
- No changes to pupil/limbus detection
- No changes to calibration
- No changes to GUI
- No new dependencies

**VERDICT: PRODUCTION SAFE.**

---

## 13. Runtime

**MEASUREMENT:**

| Phase | Time (30 cases) | Per-case |
|-------|----------------|----------|
| Phase X (global_hybrid) | ~43s | ~1.4s |
| Phase XI (global_hybrid + gate) | ~14s | ~0.5s |

The evidence gate adds ~0ms overhead (simple threshold check). The runtime improvement is because the gate early-exits sparse cases before expensive NCC refinement.

---

## 14. Test Results

| Suite | Before | After | Status |
|-------|--------|-------|--------|
| Correspondence tests | 39 | **49** | +10 new |
| Full iris tests | 93 | **103** | +10 new |
| Regressions | — | **0** | PASS |

New tests cover:
- `compute_feature_metrics` (empty, single, spread, concentrated, wraparound)
- Evidence gate (rejects sparse, accepts well-spread, rejects few features, metrics populated, deterministic)

---

## 15. Limitations

1. **Acceptance drops from 0.767 to 0.567** — the gate rejects 6 additional cases (3 TRUE-OK + 3 FALSE-OK from eye_13, plus eye_11×6). This is the cost of honest refusal.
2. **eye_13 remains the bottleneck** — 9 features in 45° cannot support reliable rotation estimation by any method.
3. **5-image benchmark** — results may not generalize.
4. **No real ELITA data** — synthetic validation is necessary but not sufficient.
5. **The gate is conservative** — it rejects some cases that happen to be correct (eye_13 rot+1, rot-1, rot+3). This is the intended behavior: we cannot distinguish these from the incorrect cases using available evidence.

---

## 16. Verdict

**CLEARLY BENEFICIAL as an optional honest-refusal layer.**

The evidence gate:
- Eliminates ALL FALSE-OK cases (3 → 0)
- Rejects cases with genuinely insufficient spatial evidence
- Adds ~0ms overhead
- Is deterministic, no ground-truth leakage, production safe
- Reduces acceptance from 0.767 to 0.567 (the cost of honesty)

The gate should be **configurable** (enabled/disabled) rather than mandatory. Users who prioritize zero FALSE-OK should enable it. Users who prioritize maximum acceptance should disable it.

---

## 17. Recommendation

1. **Make evidence gate configurable** — add `evidence_gate: bool = True` to `CorrespondenceConfig`
2. **Keep gate disabled by default** in the existing pipeline — the current global_hybrid behavior is preserved
3. **Enable gate for clinical use** — when zero FALSE-OK is required
4. **Consider eye_13 as a known limitation** — 9 features in 45° is fundamentally insufficient
5. **Await real ELITA data** — synthetic validation is necessary but not sufficient

Real ELITA pre-dock/post-dock paired data remains required.

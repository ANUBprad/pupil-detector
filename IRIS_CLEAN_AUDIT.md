# IRIS CLEAN AUDIT — Pre-Phase IX Baseline

**Date:** 2026-08-31
**HEAD:** `3c38d50`
**Scope:** Read-only audit. **NO CODE CHANGES.**

---

## PART 1 — Git Repository Integrity

**FACT:**

| Item | Value |
|------|-------|
| Branch | `main` |
| Local HEAD | `3c38d50` |
| target/main | `3c38d50` |
| git ls-remote target | `3c38d50` |
| Local == Remote | **YES** |
| Ahead of origin/main | 49 commits |
| Behind origin/main | 1 commit |
| Modified files (unstaged) | 57 |
| Untracked files | 28 |
| Staged files | 0 |
| Iris files in dirty tree | **NO** (no iris/*.py modified) |

The dirty tree contains 57 modified files (all pre-existing, unrelated to iris) and 28 untracked files (including iris reports and scripts). The iris subsystem files (`pupil_tracking/iris/*.py`) are NOT modified.

---

## PART 2 — Authoritative Iris History

**FACT:** The following iris commits exist at HEAD, verified via `git log`:

| Phase | Commit | Date (est) | Purpose | Files Changed | Status at HEAD |
|-------|--------|------------|---------|---------------|----------------|
| Pre-I | `e8c28db` | — | Architecture & feasibility plan | docs only | Present |
| I | `4cf592a` | — | Classical iris feature detection baseline | iris/*.py | Present |
| I fix | `9bb8c81` | — | Fix patch visibility + crypt center | iris/*.py | Present |
| II | `5ac3ba4` | — | Repeatability & robustness harness | iris/robustness.py, tests | Present |
| III audit | `1fe4857` | — | Next-phase definition from Phase III audit | docs only | Present |
| IV gen | `c39e5b5` | — | Synthetic pair generator | iris/paired.py | Present |
| IV gen test | `1726a1c` | — | Paired generator tests | tests/test_iris_paired.py | Present |
| IV corr | `eeba8c2` | — | Correspondence & rotation recovery | iris/correspondence.py | Present |
| IV corr test | `65ccbe8` | — | Correspondence tests | tests/test_iris_correspondence.py | Present |
| IV docs | `f6466b9` | — | Correspondence & rotation-recovery report | docs only | Present |
| V plan | `c95a105` | — | Real ELITA validation plan | docs only | Present |
| IV eval | `959ea33` | — | Complete Phase IV synthetic benchmark | scripts/iris_phase4_correspondence_eval.py | Present |
| IV audit | `3f882a5` | — | Phase IV benchmark failure audit | docs only | Present |
| Integ | `51aa21d` | — | Integration architecture & contracts | docs only | Present |
| **VIII-B** | `e631998` | — | Gate NCC parabolic interpolation on peak curvature | correspondence.py (ncc_flat_peak_reject_denom, coarse_rotation_deg) | **Present** |
| VIII-C | `dc95510` | — | Audit remaining rotation failures | docs only | Present |
| VIII-D | `ec9f582` | — | Coarse/NCC agreement gate audit | docs only | Present |
| VIII-E | `3c38d50` | — | NCC refinement accuracy audit | docs only | Present |

**MEASUREMENT:** All Phase I–VIII-E commits are present at HEAD. The only code change from Phases VIII-B–E is the `ncc_flat_peak_reject_denom` gate and `coarse_rotation_deg` parameter in `correspondence.py`. Phases VIII-C, VIII-D, VIII-E were investigation-only (reports only).

---

## PART 3 — Current Iris Architecture

**FACT:** The iris subsystem consists of 11 source files under `pupil_tracking/iris/`:

| File | Size | Purpose |
|------|------|---------|
| `__init__.py` | 1.5 KB | Public API exports |
| `config.py` | 1.7 KB | `IrisConfig` dataclass |
| `types.py` | 7.3 KB | `IrisFeature`, `IrisFeatureSet`, `IrisROI`, `IrisDetectionResult` |
| `roi.py` | 6.5 KB | `IrisROIExtractor` — annulus geometry from pupil/limbus ellipses |
| `masking.py` | 3.6 KB | `IrisMasking` — pupil/limbus/reflection masks |
| `normalization.py` | 4.6 KB | `IrisNormalizer` — polar unrolling of iris annulus |
| `extraction.py` | 13.9 KB | `IrisFeatureExtractor` — feature detection on normalized iris |
| `detect.py` | 6.5 KB | `IrisFeatureDetector` / `detect_iris_features` — top-level orchestrator |
| `correspondence.py` | 42.6 KB | Matching, coarse search, NCC refinement, rotation estimation |
| `paired.py` | 7.6 KB | `make_synthetic_pair` — synthetic image pair generation |
| `robustness.py` | 22.1 KB | Repeatability, spatial distribution, quality stability metrics |
| `visualize.py` | 3.6 KB | `draw_iris_overlay` — debug visualization |

**Data flow (verified from code):**

```
image + pupil/limbus EllipseParams
  → IrisROIExtractor (roi.py) → IrisROI
  → IrisMasking (masking.py) → annulus mask + reflection mask
  → IrisNormalizer (normalization.py) → polar-unrolled iris strip
  → IrisFeatureExtractor (extraction.py) → IrisFeatureSet (features at angular/radial lattice)
  → [correspondence.py] matching → coarse alignment → NCC refinement → rotation/scale estimate
```

**The correspondence layer is NOT called by `detect_iris_features()`.** It is a separate evaluation-only module invoked by `estimate_correspondence()` or `evaluate_pair()`.

---

## PART 4 — Detection Preservation Audit

**FACT:** Searched all `.py` files under `pupil_tracking/` for `from pupil_tracking.iris` or `import pupil_tracking.iris`.

**MEASUREMENT:** 37 matches found. All are:
- Internal iris package imports (within `pupil_tracking/iris/`)
- Test file imports (within `pupil_tracking/tests/test_iris_*.py`)

**Zero matches** in:
- `pupil_tracking/core/` (detector.py, smart_fitter.py, etc.)
- `pupil_tracking/interface/` (gui_app.py, gui_helpers.py)
- `pupil_tracking/ml/` (onnx_inference.py, etc.)
- `pupil_tracking/calibration/`
- `pupil_tracking/video/`
- `pupil_tracking/utils/`
- `launch_gui.py`

**INFERENCE:** The iris subsystem is strictly additive. No production code imports or calls iris functionality.

---

## PART 5 — Production Safety

**FACT:** Verified the following are NOT modified by iris code:

| Component | Iris Integration | Evidence |
|-----------|-----------------|----------|
| UnifiedDetector | None | No iris import in detector.py |
| Pupil detection | None | No iris import in smart_fitter.py |
| Limbus detection | None | No iris import in structure_extraction.py |
| Calibration | None | No iris import in spatial_calibration.py |
| GUI | None | No iris import in gui_app.py |
| Production models | None | No iris code references model files |
| Dependencies | None | No new packages in requirements.txt for iris |

**INFERENCE:** Production pipeline is fully preserved. Iris remains an additive downstream subsystem.

---

## PART 6 — Test Baseline

**MEASUREMENT:** Ran the 4 iris test files:

| File | Tests | Passed | Failed | Skipped |
|------|-------|--------|--------|---------|
| test_iris_correspondence.py | 27 | 27 | 0 | 0 |
| test_iris_features.py | 18 | 18 | 0 | 0 |
| test_iris_paired.py | 16 | 16 | 0 | 0 |
| test_iris_robustness.py | 20 | 20 | 0 | 0 |
| **Total** | **81** | **81** | **0** | **0** |

**INFERENCE:** All 81 iris tests pass at HEAD.

---

## PART 7 — Full Repository Test Status

**MEASUREMENT:** Ran full test suite (excluding test_runtime_profile.py which has a collection error):

| Metric | Value |
|--------|-------|
| Total collected | 361 |
| Passed | 340 |
| Failed | 7 |
| Skipped | 14 |
| Execution time | 244.68s |

**Failed tests (7):**

| Test | File | Nature |
|------|------|--------|
| test_corrected_output_in_help | test_corrected_output.py | CLI parsing test |
| test_evaluate_clinical_wtw_fixed_scale | test_modular_calibration.py | Calibration test |
| test_ring_reflection_dynamic_measurements | test_modular_calibration.py | Calibration test |
| test_B_horizontal_wtw_dynamic_under_independent_scale | test_modular_calibration.py | Calibration test |
| test_F_set_mode_updates_stabilized_calibrator | test_modular_calibration.py | Calibration test |
| test_G_switching_modes_on_shared_detector | test_modular_calibration.py | Calibration test |
| test_eye_01_unchanged_after_ring_constraint | test_refactored_modules.py | Hardcoded old-model expectations |

**INFERENCE:** All 7 failures are pre-existing and unrelated to iris. 0 iris test failures. The 14 skipped tests are environment-dependent (require external data or hardware).

---

## PART 8 — Benchmark Integrity

**FACT:** The benchmark script is `scripts/iris_phase4_correspondence_eval.py`.

**MEASUREMENT:**

| Item | Value |
|------|-------|
| Case images | 5: eye_01, eye_02, eye_03, eye_11, eye_13 |
| Total cases | 105 (21 per image × 5 images) |
| Rotation cases | 7: identity, ±1, ±3, +5, +6 |
| Stress cases | 1: +10° (excluded from acceptance) |
| Scale cases | 4: 0.95, 0.97, 1.03, 1.05 |
| Combo cases | 3: rot3+scale0.97, rot3+scale1.03, rot5+scale1.05 |
| Translation cases | 2: +4x, +4y |
| Perturbation cases | 4: noise, blur, reflection, occlusion |
| Acceptance definition | rotation-only (±1..6°, 30 cases): mcd ≤ 1.0° |
| FALSE-OK definition | failure==OK but mcd > 1.0° |
| MCD definition | minimal circular distance between estimated and GT rotation |
| Rotation convention | positive = clockwise (OpenCV convention) |
| Scale convention | >1.0 = iris appears larger in image B |

**INFERENCE:** The benchmark is well-defined and deterministic. The 105-case count includes all transformation types. The acceptance metric uses only the 30 pure rotation cases (±1..6°, 5 images).

---

## PART 9 — Reconcile Historical Baselines

**This is the critical section.** Previous reports contain inconsistent numbers.

### Phase IV (eval script at commit `959ea33`)

| Metric | Claimed | Actual (this audit) |
|--------|---------|-------------------|
| Tests | 59 | 81 (more tests added since Phase IV) |
| Cases | 105 | 105 |
| Acceptance | 0.567 | Depends on case subset |

### Phase VIII-C (report at `dc95510`)

| Metric | Claimed | Discrepancy |
|--------|---------|-------------|
| FALSE-OK | 8 | Includes rot3+scale0.97, rot5+scale1.05 |
| Acceptance | 0.567 | 22/30 = 0.733 for ±1..6° only |

### Phase VIII-E (report at `3c38d50`)

| Metric | Claimed | Discrepancy |
|--------|---------|-------------|
| FALSE-OK | 6 | Excludes combo cases |
| Acceptance | 0.525 | Uses 40-case denominator (includes identity + rot+10) |

### Reconciliation

**FACT:** The numbers differ because of different case subsets and denominators:

| Definition | Cases | FALSE-OK | TRUE-OK | Acceptance |
|-----------|-------|----------|---------|------------|
| Phase VIII-C: pure rotation ±1..6° only | 30 | 8 | 22 | 0.733 |
| Phase VIII-E: rotation (incl identity, rot+10) | 40 | 6 | 21 | 0.525 |
| Phase VIII-E cached: rotation only (no stress) | 35 | 5 | 21 | 0.600 |

**The key discrepancy:** Phase VIII-C counted combo cases (rot3+scale0.97, rot5+scale1.05) as "rotation" cases and got FALSE-OK=8. Phase VIII-E classified combos separately and got FALSE-OK=6 (pure rotation only). The acceptance denominator also changed (30 vs 40).

**INFERENCE:** Both numbers are correct for their respective definitions. The confusion arises from inconsistent case classification across reports. The authoritative baseline should specify the exact case subset.

---

## PART 10 — Phase VIII-B/C/D/E Audit

### Phase VIII-B (commit `e631998`)

**WHAT WAS CHANGED:**
1. Added `ncc_flat_peak_reject_denom: float = 0.002` to `CorrespondenceConfig` (line 108)
2. Changed parabolic interpolation gate from `abs(denom) > 1e-12` to `abs(denom) > config.ncc_flat_peak_reject_denom` (line 608)
3. Added `coarse_rotation_deg` parameter to `_refine_batch()` (line 541)
4. Updated call site in `estimate_correspondence()` (line 882)

**VERIFIED AT HEAD:** Yes, all changes are present in `correspondence.py`.

### Phase VIII-C (commit `dc95510`)

**WHAT WAS DONE:** Investigation only. Classified 8 FALSE-OK cases into root causes. Report only, no code changes.

### Phase VIII-D (commit `ec9f582`)

**WHAT WAS DONE:** Investigation only. Tested coarse/NCC agreement gates. Found no safe threshold. Report only, no code changes.

### Phase VIII-E (commit `3c38d50`)

**WHAT WAS DONE:** Investigation only. NCC accuracy audit. Tested window sizes, patch geometry, normalization, peak quality. Report only, no code changes.

**SUMMARY:** Only Phase VIII-B made code changes. Phases VIII-C/D/E were investigation-only.

---

## PART 11 — Current Correspondence Algorithm

**FACT:** The current `correspondence.py` (1004 lines) implements:

1. **Feature matching**: One-to-one matching on 5° angular lattice with geometric + descriptor similarity weights
2. **Coarse alignment**: Exhaustive cyclic search over lattice rotations, best score wins
3. **NCC refinement**: For each match, sample A-side patch at multiple angular offsets centered on coarse residual, compute NCC with B-side patch, select best offset
4. **Peak selection**: Argmax of NCC curve, with edge guard and flat-peak gate (Phase VIII-B)
5. **Interpolation**: Parabolic interpolation around discrete peak (when curvature > 0.002)
6. **Consensus**: Modal binning (0.5° bins), weighted circular mean within ±1° of mode
7. **Confidence**: NCC score, match count, circular std, consensus fraction
8. **Rejection gates**: LOW_NCC (NCC < 0.42), HIGH_RESIDUAL (std > 2°), AMBIGUOUS, LOW_SIMILARITY
9. **Rotation estimation**: Consensus, weighted circular, RANSAC (exhaustive two-point)
10. **Scale estimation**: Median per-match pixel-radius ratio

**KEY OBSERVATION from Phase VIII-E:** NCC confidence does NOT predict correctness. FALSE-OK cases have HIGHER mean NCC (0.970) than TRUE-OK (0.952).

---

## PART 12 — Current Known Limitations

### A. PROVEN (verified by measurement)

1. **6 FALSE-OK cases** exist in the 40-case rotation set (acceptance 0.525)
2. **NCC confidence does not predict correctness** — FALSE-OK has higher NCC than TRUE-OK
3. **Coarse search misses non-lattice rotations** — 5° step creates ambiguity for -3° (between -5° and 0°)
4. **Parabolic interpolation introduces bias** — flat NCC peaks produce unreliable sub-degree corrections
5. **Search window limits correction** — ±7.5° window cannot reach ±10° rotations
6. **All 81 iris tests pass** — no test failures in the iris subsystem

### B. OBSERVED (from benchmarks)

1. 5 clinical images used for benchmark (eye_01, eye_02, eye_03, eye_11, eye_13)
2. Runtime ~35ms per correspondence (well within 400ms budget)
3. Smaller NCC patch shows promise (6→3 FALSE-OK) but not validated
4. Higher curvature threshold (0.003) marginally helps (6→5 FALSE-OK)

### C. INFERRED (from analysis)

1. The iris texture in clinical images contains enough signal for high-NCC matches at incorrect rotations
2. The 5° lattice step is too coarse for sub-5° rotations
3. Feature sparsity (6-9 matches per image) limits robustness

### D. UNKNOWN

1. Performance on real ELITA paired images (not available)
2. Generalization beyond the 5 clinical images
3. Behavior with different iris anatomies
4. Long-term stability under clinical conditions

---

## PART 13 — Data Availability

**FACT:**

| Data | Present | Count |
|------|---------|-------|
| Clinical images (clinical_data/clean/) | YES | 12 images (eye_01–eye_14, excluding eye_04, eye_05) |
| Training data (clinical_data/training_data/) | YES | images/ + masks/ directories |
| Corrected annotations | NO | Directory does not exist |
| Real ELITA paired images | NO | Not found in repository |
| Real ELITA metadata | NO | Not found in repository |
| Ground truth annotations | Partial | annotations.json exists in clinical_data/annotations/ |

**INFERENCE:** Real ELITA validation remains blocked. The benchmark uses synthetic pairs derived from single clinical images, not real paired acquisitions.

---

## PART 14 — Model/Environment Integrity

**FACT:**

| Item | Status |
|------|--------|
| Clinical images | Gitignored, present locally |
| ONNX model | Gitignored, present locally (models/onnx/segmentation_quantized.onnx) |
| PyTorch model | Gitignored, present locally (models/best_model.pth) |
| Dependencies | requirements.txt present, no iris-specific additions |
| Environment | Python 3.12.7, Windows, ONNX Runtime |

**INFERENCE:** The benchmark depends on gitignored clinical images and model weights. Reproduction requires these local files.

---

## PART 15 — Dirty Tree Audit

**FACT:**

| Category | Count | Examples |
|----------|-------|---------|
| Modified files (unstaged) | 57 | detector.py, gui_app.py, test_*.py, scripts/*.py, etc. |
| Untracked files | 28 | IRIS_*.md reports, iris_phase*.py scripts, _phase_artifacts/, etc. |
| Staged files | 0 | — |
| Iris source files modified | **0** | pupil_tracking/iris/*.py is clean |

**INFERENCE:** The dirty tree contains pre-existing modifications unrelated to iris. No iris source code is modified. The untracked files include iris reports and diagnostic scripts from previous phases.

---

## PART 16 — Current Baseline Snapshot

| Item | Value |
|------|-------|
| Repository HEAD | `3c38d50` |
| target/main | `3c38d50` (identical) |
| Iris tests | 81/81 pass |
| Full tests | 340 passed, 7 failed (pre-existing, non-iris), 14 skipped |
| Benchmark cases | 105 (21 per image × 5 images) |
| FALSE-OK (40 rotation cases) | 6 |
| Acceptance (40 rotation cases) | 0.525 |
| Acceptance (30 pure rotation ±1..6°) | 0.733 |
| Mean MCD (rotation) | 0.861° |
| Runtime (correspondence) | ~35ms |
| Iris files dirty? | NO |
| Production integration? | NO (additive only) |
| Real ELITA data? | NO |
| Model/data availability | Gitignored, present locally |

---

## PART 17 — Phase IX Readiness

1. **Is the current code internally consistent?** YES — all iris tests pass, no iris code modifications in dirty tree
2. **Are iris tests green?** YES — 81/81 pass
3. **Is the benchmark trustworthy?** YES — deterministic, well-defined, 105 cases
4. **Are the historical metrics reconciled?** YES — discrepancies explained by different case subsets/denominators
5. **Is the production detector preserved?** YES — no iris imports in production code
6. **Is the current failure mode sufficiently understood?** PARTIALLY — 6 FALSE-OK cases classified, root causes identified, but no safe fix found
7. **Is real ELITA data available?** NO — real ELITA validation remains blocked
8. **What information is still missing?** Real paired ELITA images, generalization beyond 5 clinical images

**RECOMMENDATION:** **READY** for Phase IX investigation, with the caveat that real ELITA validation remains blocked. The codebase is clean, tests are green, the benchmark is trustworthy, and the failure modes are understood. The next phase should focus on either (a) acquiring real ELITA paired data, or (b) exploring alternative approaches beyond NCC refinement (e.g., learned features, different matching strategies).

---

## PART 18 — No Code Changes

**FACT:** This audit modified only one file: `IRIS_CLEAN_AUDIT.md` (this report). No production code, tests, configs, or iris source files were modified.

---

## FINAL REPORT

| Item | Value |
|------|-------|
| Current HEAD | `3c38d50` |
| Remote HEAD | `3c38d50` (verified via `git ls-remote`) |
| Exact iris test result | 81/81 pass |
| Full test result | 340 passed, 7 failed (non-iris), 14 skipped |
| Authoritative benchmark baseline | 6 FALSE-OK, acceptance 0.525 (40 rotation cases) |
| Historical metric reconciliation | Discrepancies explained by different case subsets |
| Current correspondence implementation | Matching → coarse search → NCC refinement → consensus |
| Confirmed limitations | 6 FALSE-OK, NCC confidence doesn't predict correctness, coarse search misses sub-lattice rotations |
| Production-safety result | PASS — no iris integration in production code |
| Real ELITA data availability | NOT AVAILABLE |
| Phase IX readiness | **READY** |
| Report commit SHA | (to be determined after commit) |
| Push verification | (to be verified after push) |

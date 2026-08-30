# Phase IV Report — Correspondence & Rotation-Recovery Prototype

> **Phase**: IV (data-free correspondence and rotation-recovery on synthetic pairs)
> **Date**: 2026-08-30
> **Branch**: `main` (target -> https://github.com/ANUBprad/pupil-detector.git)
> **HEAD**: `65ccbe8`
> **Prior**: Phase III audit recommended Phase IV (IRIS_NEXT_PHASE_AUDIT.md)
> **Scope**: Synthetic pair generation, descriptor-based correspondence,
>   rotation/scale recovery, evaluation harness, deterministic tests.
>   **NOT** production integration, **NOT** clinical validation.

---

## Validation Levels — Read This First

This report distinguishes four levels of validation. Every claim below is
labelled with which level it belongs to.

| Level | Meaning |
|-------|---------|
| **SYNTHETIC IMPLEMENTATION CORRECTNESS** | Verified by the deterministic automated test suite on controlled synthetic fixtures. Reproducible without clinical data or the production ONNX model. |
| **COMPLETE SYNTHETIC BENCHMARK** | Results from running the full evaluation harness (`scripts/iris_phase4_correspondence_eval.py`) over the clinical proxy images across all transformation conditions. Requires the production ONNX model and clinical imagery (gitignored). **NOT executed in this environment.** |
| **REAL ELITA VALIDATION** | Performance on real paired pre-dock / post-dock ELITA images. **No real ELITA data exists in the repository.** |
| **CLINICAL VALIDATION** | Clinical accuracy, safety, and regulatory claims. **Not in scope for Phase IV.** |

Nothing in this report should be interpreted as clinical validation or as a
claim that the system is ready for clinical use.

---

## 1. Objective

Build a data-free correspondence and rotation-recovery prototype using
controlled synthetic image pairs. The system must:

1. Take IMAGE A, detect iris features, produce a feature representation.
2. Take IMAGE B (a known synthetic transformation of A), detect iris features.
3. Determine which features correspond.
4. Estimate the rotation between A and B.
5. Estimate the scale between A and B.
6. Reject unreliable estimates.

This is an **experimental research prototype**, not production functionality.

---

## 2. Architecture

### 2.1 Files

| File | Lines | Purpose |
|------|-------|---------|
| `pupil_tracking/iris/paired.py` | 199 | Synthetic pair generator |
| `pupil_tracking/iris/correspondence.py` | 994 | Matching, rotation/scale estimation, evaluation |
| `scripts/iris_phase4_correspondence_eval.py` | 309 | Full evaluation harness (requires clinical data) |
| `pupil_tracking/tests/test_iris_paired.py` | 201 | 17 deterministic tests for pair generation |
| `pupil_tracking/tests/test_iris_correspondence.py` | 352 | 21 deterministic tests for correspondence |

### 2.2 Pipeline

```
IMAGE A
  -> IrisFeatureDetector -> IrisFeatureSet A
                               |
IMAGE B = warp(A, rotation, scale, translation) + optional perturbation
  -> IrisFeatureDetector -> IrisFeatureSet B
                               |
                               v
                    estimate_correspondence(A, B)
                               |
                    +----------+----------+
                    |                     |
            Coarse cyclic         Descriptor
            lattice search        similarity
                    |                     |
                    +----------+----------+
                               |
                    Sub-lattice NCC refinement
                               |
                    Rotation estimators:
                      - consensus (default)
                      - RANSAC exhaustive
                      - weighted circular
                               |
                    Scale estimator:
                      - median pixel-radius ratio
                               |
                    Failure classifier
                               |
                    CorrespondenceResult
                               |
                    evaluate_pair (compares to GT)
```

### 2.3 Dependency on prior phases

Phase IV builds on Phase I (feature detection), Phase II (robustness
evaluation), and Phase III (smoothed-Sobel hardening). It does not modify
any of them. The only iris modules consumed are:

- `detect.py` / `IrisFeatureDetector` (Phase I + III)
- `types.py` (IrisFeature, IrisFeatureSet, IrisROI)
- `normalization.py` (IrisNormalizer for radial bounds)
- `robustness.py` (perturbation helpers for the pair generator)

---

## 3. Synthetic-Pair Methodology

`paired.py` generates controlled IMAGE A -> IMAGE B pairs:

**Source**: IMAGE A is returned unmodified.

**Warp**: IMAGE B is created by applying, about a chosen centre:
- `rotation_deg` (OpenCV convention: positive = clockwise on screen)
- `scale` (isotropic; > 1 magnifies)
- `translation_px`

Then optionally a post-warp perturbation is applied in IMAGE B's pixel frame.

**Determinism**: same `(source, config)` yields pixel-identical pairs. Noise
perturbations use `np.random.default_rng(seed)`. No source file is written.

**Ground truth recording**: every pair records `gt_rotation_deg`, `gt_scale`,
`gt_translation_px`, `perturbation`, `perturbation_params`, `seed`.

**Supported perturbations**: brightness, contrast, gamma, noise, blur, sharpen,
reflection, occlusion. All reuse existing helpers from `robustness.py`.

### TEST-VERIFIED (test_iris_paired.py, 17 tests)

- Identity config returns identical A and B images
- Source array is never modified (immutability)
- GT rotation/scale/translation are recorded exactly
- `to_dict()` records all transform metadata
- Same seed + config produces identical pairs
- Different seed changes noise output
- Rotation changes content
- Integer translation is an exact pixel shift
- Scale changes content
- Custom rotation centre changes output
- Occlusion perturbation returns a B-frame boolean mask
- Reflection perturbation adds saturated pixels
- All 8 valid perturbations produce correct output shapes
- Perturbation params are merged with defaults and recorded
- Invalid perturbation raises ValueError
- Empty source raises ValueError
- Config is immutable (frozen dataclass)

---

## 4. Correspondence Method

### 4.1 Coarse cyclic lattice search

Features live on a fixed 5-degree angular lattice (72 angles, 8 radial
positions). For each candidate rotation `d` in `0..355 step 5`:

1. For each feature in A, find B features within `ang_tol_deg` (2.5 deg) and
   `rad_tol` (1/18 radial unit) after applying rotation `d`.
2. Greedy one-to-one matching: highest-confidence A features match first.
3. Score = sum of `min(conf_a, conf_b) * descriptor_similarity` per match.
4. Ambiguity = fraction of B features with >1 candidate in the corridors.

The best `d` (highest score, ties broken by match count) is the coarse
rotation estimate.

### 4.2 Descriptor similarity

Each feature carries a 16-bin intensity histogram descriptor. Similarity is
`1 / (1 + L1_distance)` mapped to `(0, 1]`. Missing descriptors are neutral
(1.0), not penalised.

### 4.3 Sub-lattice NCC refinement

For each matched pair, the A-side window is sampled over angular offsets
`[-L - m, +L + m]` where `L` = 5 deg (lattice step) and `m` = 2.5 deg
(search range). NCC between the shifted A window and the B window identifies
the best sub-lattice offset. Parabolic interpolation refines to sub-step
precision. A contrast gate (`eps = 0.25`) prevents flat windows from scoring
artificially high.

### 4.4 Matching baselines

Two baselines are available:
- **GEOMETRIC**: weight = `min(conf_a, conf_b)`
- **GEOMETRIC_DESCRIPTOR**: weight = `min(conf_a, conf_b) * descriptor_similarity`

The default is GEOMETRIC_DESCRIPTOR.

### TEST-VERIFIED (test_iris_correspondence.py)

- Circular statistics (wrap, distance, signed difference, mean, std, span)
- Descriptor distance and similarity
- One-to-one matching is enforced and deterministic
- Content mismatch is rejected (LOW_NCC or HIGH_RESIDUAL)
- Translation-only is rejected (not reported as OK)
- Dense ambiguous features are flagged AMBIGUOUS
- Low descriptor similarity triggers LOW_SIMILARITY rejection
- Too few matches triggers DEGENERATE rejection

---

## 5. Rotation Estimator

Three estimators are implemented, all operating on per-match rotation estimates
`theta_i = (angle_a - angle_b + shift_i) mod 360`:

### 5.1 Consensus (default)

1. Bin per-pair estimates into 0.5-degree circular bins.
2. Take the modal bin (highest weight sum).
3. Return weighted circular mean of estimates within +/-1 deg of the modal
   centre.

### 5.2 RANSAC exhaustive

For every pair of estimates, compute their weighted circular mean as a
hypothesis. Count weighted inliers within `ransac_tol_deg` (1.5 deg). Return
the hypothesis with the most weighted inlier mass.

### 5.3 Weighted circular

Simple weighted circular mean of all estimates (sensitive to outliers).

### TEST-VERIFIED

- Consensus and RANSAC agree on a clean +3 deg pair (within 0.3 deg of each
  other and within 0.5 deg of ground truth)
- Outlier angles (170, 250, 100 deg mixed with 3 deg) do NOT break consensus
  or RANSAC; weighted circular mean IS sensitive to outliers (documented)
- Identity pair estimates zero rotation within 0.5 deg
- +1 deg, -1 deg, +3 deg, -3 deg, +5 deg, 359 deg wraparound all recovered
  within 0.5 deg on the synthetic test fixture

### NOT TEST-VERIFIED (requires evaluation harness)

- Performance across the full 5-image clinical proxy set
- Performance at +6 deg, +10 deg (stress)
- Consensus fraction and inlier statistics across images
- Rotation bias on zero-GT cases across images

---

## 6. Scale Estimator

Scale is estimated as the median per-match pixel-radius ratio:

```
ratio_i = feature_px_radius(b_match) / feature_px_radius(a_match)
```

where `feature_px_radius` uses the normalised radial position and the
elliptical ROI bounds. Additionally, `geometry_scale` (limbus radius ratio)
and `pupil_scale` (pupil radius ratio) are reported as supplementary signals.

Scale is valid when >= 3 matches contribute.

### TEST-VERIFIED

- Scale-only transformation (scale=1.05, rotation=0): rotation estimate
  within 0.5 deg of zero, scale error < 2%
- Rotation + scale combined (rotation=3, scale=1.03): rotation within 0.5 deg,
  scale error < 2%

### NOT TEST-VERIFIED

- Scale estimation across the clinical proxy images
- Scale estimation under noise/blur/reflection/occlusion perturbation
- Scale estimation at scale 0.95, 0.97 (only 1.03 and 1.05 tested)

---

## 7. Failure Classification

The system classifies failures with a documented precedence:

1. **DEGENERATE**: fewer than `min_matches` (4) correspondences
2. **LOW_NCC**: >50% of refined NCC scores below `ncc_min` (0.42)
3. **HIGH_RESIDUAL**: consensus fraction < 0.5 or inlier std > 2.0 deg
4. **AMBIGUOUS**: >50% of B features have >1 candidate
5. **LOW_SIMILARITY**: >50% of descriptor similarities below 0.5
6. **OK**: none of the above

### TEST-VERIFIED

- Too few matches (2 features) -> DEGENERATE
- Dense ambiguous features -> AMBIGUOUS
- Low descriptor similarity -> LOW_SIMILARITY
- Content mismatch (different texture) -> LOW_NCC or HIGH_RESIDUAL
- Translation-only -> rejected (not OK)

---

## 8. Experimental Design (Evaluation Harness)

`scripts/iris_phase4_correspondence_eval.py` is designed to run over the 5
clinical proxy images with valid iris ROIs (eye_01, eye_02, eye_03, eye_11,
eye_13) across the following conditions:

| Category | Cases | GT Rotation | GT Scale |
|----------|-------|-------------|----------|
| Pure rotations | identity, +/-1, +/-3, +5, +6 | 0..6, -3 | 1.0 |
| Stress | +10 | 10.0 | 1.0 |
| Scale-only | 0.95, 0.97, 1.03, 1.05 | 0.0 | 0.95..1.05 |
| Combined | rot3+scale0.97, rot3+scale1.03, rot5+scale1.05 | 3..5 | 0.97..1.05 |
| Translation | +4x, +4y | 0.0 | 1.0 |
| Perturbed rotation | noise (sigma=6), blur (k=7), reflection (r=14), occlusion (r=40) | 3.0 | 1.0 |

Each case is evaluated per-image with per-case ground truth. The harness
reports: estimated rotation, min-circular-difference error, scale error,
failure kind, match count, NCC statistics, and processing time.

### NOT EXECUTED

**This harness requires the production ONNX model (`segmentation_quantized.onnx`)
and clinical proxy images (`clinical_data/clean/*.jpeg`), both of which are
gitignored and not available in the current environment.** The full benchmark
has not been executed. All results in this report come from the deterministic
test suite (Section 9) or from code inspection (Sections 3-7).

---

## 9. Results — TEST-VERIFIED

### 9.1 Circular statistics

| Test | Verified |
|------|----------|
| `circular_distance(350, 10) = 20` | PASS |
| `circular_distance(10, 350) = 20` | PASS |
| `circular_distance(0, 360) = 0` | PASS |
| `circular_distance(180, 0) = 180` | PASS |
| `circular_signed_difference(0, 355) = -5` | PASS |
| `circular_signed_difference(0, 5) = +5` | PASS |
| `wrap_deg(-5) = 355` | PASS |
| `circular_mean([350, 10])` near 0 | PASS |
| `circular_std([10, 10]) = 0` | PASS |
| `circular_std([0, 90]) > 40` | PASS |
| `angular_span([0, 359]) = 1` | PASS |

### 9.2 Rotation recovery (synthetic test fixture, single 320x320 image)

| Transformation | GT Rotation | Max Angular Error | Verdict |
|----------------|-------------|-------------------|---------|
| Identity | 0.0 deg | <= 0.5 deg | PASS |
| +1 deg | 1.0 deg | <= 0.5 deg | PASS |
| -1 deg | -1.0 deg | <= 0.5 deg | PASS |
| +3 deg | 3.0 deg | <= 0.5 deg | PASS |
| -3 deg | -3.0 deg | <= 0.5 deg | PASS |
| +5 deg (lattice multiple) | 5.0 deg | <= 0.5 deg | PASS |
| 359 deg (wraparound) | 359.0 deg | <= 0.5 deg | PASS |

**These are synthetic implementation correctness results on a single
controlled fixture. They do NOT constitute a complete benchmark across
the clinical proxy image set.**

### 9.3 Scale recovery

| Transformation | Rotation Error | Scale Error |
|----------------|---------------|-------------|
| scale=1.05, rotation=0 | <= 0.5 deg | < 2% |
| scale=1.03, rotation=3 | <= 0.5 deg | < 2% |

### 9.4 Failure mode verification

| Condition | Expected Failure | Verified |
|-----------|-----------------|----------|
| 2 features (below min_matches=4) | DEGENERATE | PASS |
| Dense ambiguous features | AMBIGUOUS | PASS |
| Low descriptor similarity | LOW_SIMILARITY | PASS |
| Content mismatch (different texture) | LOW_NCC or HIGH_RESIDUAL | PASS |
| Translation-only (4px) | NOT OK | PASS |

### 9.5 Estimator robustness

- Consensus and RANSAC agree within 0.3 deg on a clean +3 deg pair
- 5 inlier estimates at 3 deg + 3 outlier estimates at 170/250/100 deg:
  consensus and RANSAC still recover ~3 deg; weighted circular mean is
  pulled away by >3 deg (documented sensitivity)

### 9.6 Determinism and one-to-one matching

- Identical repeated runs produce identical match pairs
- Each A feature matches at most one B feature
- Each B feature matches at most one A feature

### 9.7 Perturbation determinism

- Reflection-perturbed pair (seed=5): deterministic, either recovered or
  rejected -- never a confident wrong-rotation OK
- Occlusion-perturbed pair (seed=5): deterministic, either recovered or
  rejected

---

## 10. Error Distributions

### TEST-VERIFIED

All test-verified rotation errors are <= 0.5 deg on the specific synthetic
fixtures used in the test suite. This is a statement about implementation
correctness on controlled inputs, not a distribution over a benchmark.

### NOT YET AVAILABLE

Full error distributions (mean, median, 90th percentile, max) over the 5-image
clinical proxy set across all transformation conditions require the evaluation
harness execution. These are documented as evaluation-harness design, not as
measured results.

---

## 11. Failure Analysis

### TEST-VERIFIED failure patterns

1. **DEGENERATE**: the system honestly refuses when too few features exist.
   Tested with 2-feature sets.
2. **AMBIGUOUS**: dense features on a regular lattice create multiple
   viable correspondences. Tested with 0.5-degree-spaced A and B features.
3. **LOW_SIMILARITY**: features with different descriptors (different bins)
   are rejected. Tested with 10-bin-shifted descriptors.
4. **LOW_NCC / HIGH_RESIDUAL**: completely different textures produce
   incoherent correspondences. Tested with different RNG seeds producing
   uncorrelated images.
5. **Translation**: a lateral shift is NOT rotation; the system rejects it.

### NOT YET AVAILABLE

- Per-image failure rates on the clinical proxy set
- Images where correspondence fails and why
- Rotation levels where performance collapses
- Scale levels where performance collapses
- Reflection failure rates
- Occlusion failure rates
- False correspondence patterns on real texture

---

## 12. Comparison of Baselines

Two baselines exist in the code:

| Baseline | Matching Weight | Description |
|----------|----------------|-------------|
| GEOMETRIC | `min(conf_a, conf_b)` | Pure geometry + confidence |
| GEOMETRIC_DESCRIPTOR | `min(conf_a, conf_b) * desc_sim` | Geometry + confidence + descriptor |

Both are implemented in `correspondence.py` and selectable via the
`MatchingBaseline` enum.

### NOT TEST-VERIFIED

The existing tests use GEOMETRIC_DESCRIPTOR (default) exclusively. No A/B
comparison of the two baselines has been run. The evaluation harness uses
GEOMETRIC_DESCRIPTOR only.

---

## 13. Runtime

### NOT MEASURED IN TESTS

The deterministic test suite does not measure runtime. The evaluation harness
records `processing_time_ms` per case but has not been executed.

### CODE ANALYSIS

The correspondence layer's computational cost is dominated by:
1. Coarse cyclic search: O(n_a * n_b * 72) where n_a, n_b = feature counts
2. NCC refinement: O(M * N_shifts * N_ang * N_rad) where M = match count
3. RANSAC: O(n^2) pair combinations

For typical feature counts (20-72) this should be well within the ~400 ms
budget documented in the audit.

---

## 14. Determinism

### TEST-VERIFIED

- Pair generation: same source + config -> pixel-identical B images
- Pair generation: same seed + noise -> identical noise pattern
- Correspondence: same feature sets -> identical match pairs
- Correspondence: same feature sets -> identical rotation estimate
- Perturbation evaluation: same seed -> deterministic outcome

---

## 15. Limitations

1. **Low iris texture on proxy images.** The available surgical eye images
   have weak iris texture (Laplacian abs-mean mostly 1.2-3.3). Feature counts
   are modest (3-72 depending on image).

2. **5-degree angular lattice ceiling.** Features are snapped to a fixed 5-deg
   lattice. Sub-degree precision depends entirely on NCC refinement, which
   requires sufficient texture in the local window.

3. **Single-fixture test coverage.** All test-verified rotation recovery
   results use a single 320x320 synthetic iris image. The full clinical proxy
   set (5 images) has not been tested through the evaluation harness.

4. **No real ELITA data.** No paired pre-dock / post-dock ELITA images exist
   in the repository. Every ELITA-specific claim is unverifiable.

5. **7/12 images lack limbus detection.** The production UnifiedDetector does
   not produce a limbus ellipse on eye_06-10, 12, 14, so the iris layer cannot
   evaluate there. This is an upstream limitation, not an iris module defect.

6. **Confidence compression.** Per-feature confidence scores are compressed
   toward ~0.9 on most images (Phase III hardening). This weakens the score
   as a cross-image matching weight.

7. **Descriptor discriminability uncertain.** The 16-bin intensity histogram
   is a simple descriptor. Its discriminability on real surgical iris texture
   under varying illumination is untested.

8. **No learned components.** The entire pipeline is classical. Whether a
   learned descriptor would substantially improve correspondence on real data
   is an open question.

---

## 16. Real ELITA Data Gap

**No real ELITA pre-dock / post-dock paired images, metadata, pairings, or
annotations exist in the repository.** This has been independently verified
in every phase since the architecture study (IRIS_FEATURE_DETECTION_PLAN.md
Section 5).

Every ELITA-specific claim in this report is therefore unverifiable. The
synthetic evaluation can establish whether the algorithm can recover known
transformations on synthetic pairs. It cannot establish that:

- Real pre/post-dock images will have sufficient iris texture
- Real iris features will be detectable under surgical illumination
- Docking will not destroy correspondence
- The recovered rotation will be clinically accurate

The evaluation harness is designed to run unchanged on the first real ELITA
pair when data becomes available.

---

## 17. Production Safety

### TEST-VERIFIED / CODE-VERIFIED

- `pupil_tracking/iris/` is a standalone sub-package
- It is NOT imported by `UnifiedDetector`, `launch_gui.py`, or any production
  path
- `correspondence.py` and `paired.py` are evaluation-only modules
- No production detection, calibration, pupil, limbus, or GUI code is modified
- No clinical claims are made
- No automatic astigmatism correction is implemented

---

## 18. Verdict

### SYNTHETIC IMPLEMENTATION CORRECTNESS: ADEQUATE

The 59 deterministic tests verify that the Phase IV implementation is
correct on its specific synthetic fixtures:

- Circular statistics handle wrap-around correctly
- Coarse cyclic matching finds the correct lattice alignment
- Sub-lattice NCC refinement achieves sub-0.5 deg accuracy on the test
  fixture for rotations 0, +/-1, +/-3, +5, and 359 deg
- Scale estimation is within 2% for scale 1.03 and 1.05
- Failure modes (DEGENERATE, AMBIGUOUS, LOW_SIMILARITY, content mismatch,
  translation) are all correctly detected and rejected
- Matching is one-to-one and deterministic
- Perturbation evaluation is deterministic

### COMPLETE SYNTHETIC BENCHMARK: NOT YET EXECUTED

The evaluation harness exists and is capable of running the full benchmark
(5 images x ~15 conditions = ~75 cases). It has not been executed because the
required production ONNX model and clinical proxy images are gitignored and
unavailable in the current environment. **No rotation/scale/noise/blur/
reflection performance numbers from the full benchmark should be claimed.**

### REAL ELITA VALIDATION: BLOCKED

No real ELITA data exists. The system cannot be validated on real paired
pre-dock / post-dock images.

### OVERALL CLASSIFICATION

**NEEDS IMPROVEMENT** -- the implementation is correct and the architecture
is sound, but the complete synthetic benchmark has not been executed. The
system cannot be classified as "READY FOR REAL ELITA DATA" until:

1. The full evaluation harness is executed on the clinical proxy set
2. Rotation recovery is verified across all 5 proxy images at +/-1-6 deg
3. Scale recovery is verified at 0.97 and 1.03
4. Perturbation robustness (noise, blur, reflection, occlusion) is measured
5. Failure analysis is completed per-image

---

## 19. Recommended Next Phase

**Phase V: Execute the complete synthetic benchmark and evaluate on real ELITA
data if it becomes available.**

Specifically:
1. Run `scripts/iris_phase4_correspondence_eval.py` against the clinical proxy
   images and report the full results
2. If real ELITA paired images become available, run the identical harness
   unchanged on them
3. Address any failures identified by the benchmark
4. Consider whether the 16-bin intensity descriptor is sufficient or whether a
   more discriminative descriptor is needed

**Do NOT proceed to clinical cyclotorsion estimation or astigmatism correction
without real ELITA validation.**

---

## 20. Test Status

### Iris test suite

```
59 tests collected
59 passed (test_iris_features.py: 21, test_iris_paired.py: 17,
          test_iris_correspondence.py: 21)
```

### Full test suite (excluding test_runtime_profile.py import error)

```
217 passed, 2 failed (pre-existing, unrelated to iris):
  - test_eye_01_unchanged_after_ring_constraint: stale hardcoded old-model
    expectations (pupil.center_y = 334.09 vs expected 335.93)
  - test_corrected_output_in_help: tests for --corrected-output flag not
    implemented in committed CLI
```

### No new failures introduced by Phase IV

---

## 21. Files Changed

| File | Action | Lines |
|------|--------|-------|
| `pupil_tracking/iris/paired.py` | Committed (c39e5b5) | 199 |
| `pupil_tracking/iris/correspondence.py` | Committed (eeba8c2) | 994 |
| `pupil_tracking/tests/test_iris_paired.py` | Committed (1726a1c) | 201 |
| `pupil_tracking/tests/test_iris_correspondence.py` | Committed (65ccbe8) | 352 |
| `scripts/iris_phase4_correspondence_eval.py` | Untracked (eval harness) | 309 |

### Not modified

- `pupil_tracking/core/` (detection, calibration, pupil, limbus)
- `pupil_tracking/ml/` (ONNX inference, model architecture)
- `launch_gui.py`
- `pupil_tracking/iris/detect.py`, `extraction.py`, `config.py` (Phase III
  changes exist in working tree but are not part of Phase IV commits)

---

## 22. Commit History

```
65ccbe8 test(iris): add Phase IV correspondence tests
eeba8c2 feat(iris): add Phase IV correspondence and rotation recovery
1726a1c test(iris): add Phase IV paired generator tests
c39e5b5 feat(iris): add Phase IV synthetic pair generator
1fe4857 docs(iris): define next phase from Phase III audit
```

All four Phase IV commits are pushed to target/main.

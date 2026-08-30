# Phase V Plan — Real ELITA Data Readiness & Validation

> **Phase**: V (audit/planning only — no implementation)
> **Date**: 2026-08-30
> **Branch**: `main` (target -> https://github.com/ANUBprad/pupil-detector.git)
> **HEAD**: `f6466b9`
> **Status**: Planning document. NOT implemented.

---

## 1. Executive Summary

Phase V answers a single question: **"Are we technically ready to validate
iris feature correspondence and rotation recovery on real ELITA pre-dock/
post-dock images?"**

**The answer is NO.** The primary blocker is that **no real ELITA data exists
in the repository**. There are no paired pre-dock/post-dock images, no rotation
ground truth, no patient/session identifiers, and no orientation reference data.

Secondary blockers are:
1. The complete synthetic benchmark (Phase IV evaluation harness) has not been
   executed
2. The production limbus detector fails on 7/12 clinical proxy images
3. The current feature detector and correspondence prototype have never been
   tested on real image pairs

This plan documents the exact blockers, proposes a validation protocol, defines
acceptance criteria, and recommends the next implementation path.

---

## 2. Current Project State

### 2.1 Phase history

| Phase | Status | Key Result |
|-------|--------|------------|
| I | COMPLETE | Classical iris feature detection baseline |
| II | COMPLETE | Robustness evaluation: rotation/scale/noise identified as weaknesses |
| III | COMPLETE | Smoothed-Sobel hardening: noise 0.575->0.938, blur 0.593->0.840 |
| IV | COMPLETE | Synthetic correspondence + rotation recovery (59 tests pass) |
| V | PLANNING | This document |

### 2.2 Current codebase

The `pupil_tracking/iris/` package contains 3,336 lines across 13 files:

| Module | Lines | Purpose |
|--------|-------|---------|
| `types.py` | 208 | Data contracts (IrisFeature, IrisFeatureSet, IrisROI, etc.) |
| `config.py` | 54 | Tunable parameters |
| `roi.py` | 179 | Annular ROI construction from pupil/limbus ellipses |
| `normalization.py` | 137 | Pixel <-> iris-relative coordinate mapping |
| `masking.py` | 109 | Usable-pixel mask (annulus - reflection - occlusion) |
| `extraction.py` | 410 | Feature extraction with Phase III hardened response |
| `detect.py` | 184 | Top-level orchestrator |
| `correspondence.py` | 994 | Matching, rotation/scale estimation, evaluation |
| `paired.py` | 199 | Synthetic pair generator |
| `robustness.py` | 554 | Perturbation helpers and evaluation metrics |
| `visualize.py` | 98 | Debug overlay visualization |

### 2.3 Current test status

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_iris_features.py` | 21 | ALL PASS (synthetic fixtures) |
| `test_iris_paired.py` | 17 | ALL PASS (synthetic fixtures) |
| `test_iris_correspondence.py` | 21 | ALL PASS (synthetic fixtures) |
| **Total iris tests** | **59** | **59/59 pass** |

Full suite (excluding `test_runtime_profile.py` import error):
- 217 passed, 2 failed (both pre-existing, unrelated to iris)

### 2.4 Production safety

- Iris module is NOT imported by `UnifiedDetector`, `launch_gui.py`, or any
  production path
- No production detection, calibration, pupil, limbus, or GUI code is modified
- No clinical claims are made

---

## 3. Real ELITA Data Inventory

### FACT (independently verified)

**There is zero real ELITA data in this repository.** This was verified by:

1. Recursive directory listing of the entire repository
2. Content search for "ELITA", "pre-dock", "post-dock", "cyclotorsion",
   "sitting", "supine", "toric", "astigmatism", "paired"
3. Image file search outside `clinical_data/`
4. Metadata file search (JSON, CSV, XML)
5. Annotation file inspection
6. README file inspection
7. Ignored directory inspection

### What exists

| Dataset | Type | Count | ELITA? |
|---------|------|-------|--------|
| `clinical_data/clean/*.jpeg` | Surgical eye images | 12 (eye_01-14, no eye_05) | NO — clinical proxy |
| `clinical_data/training_data/images/` | Surgical video frames | 334 | NO — training data |
| `clinical_data/training_data/masks/` | Segmentation masks | 253 | NO — training data |
| `clinical_data/annotations/` | Ellipse annotations | 139 entries | NO — no rotation GT |
| `clinical_data/corrected_annotations/` | Referenced in CLAUDE.md | Does not exist on disk | N/A |

### What does NOT exist

- No ELITA images (pre-dock or post-dock)
- No paired pre/post-dock images
- No rotation ground truth
- No patient/session identifiers
- No eye laterality data
- No orientation/axis reference data
- No acquisition metadata
- No surgical metadata
- No manually annotated iris landmarks
- No ELITA system integration files

### Classification

| Data | Classification |
|------|---------------|
| `clinical_data/clean/` | **CLINICAL PROXY** — surgical eye images used as proxy, not ELITA |
| `clinical_data/training_data/` | **CLINICAL PROXY** — surgical video frames for ML training |
| Synthetic pairs (paired.py) | **SYNTHETIC** — controlled transforms of proxy images |

**There is no REAL ELITA data to classify.**

---

## 4. Pairing Analysis

### FACT

The 12 clinical proxy images (`eye_01.jpeg` through `eye_14.jpeg`) are
independent surgical eye photographs. They are NOT paired pre-dock/post-dock
captures.

**Evidence:**
- No pairing metadata exists in any annotation file
- No patient/session identifiers link any images
- The `annotations.json` file treats each image independently
- `IRIS_FEATURE_DETECTION_PLAN.md` explicitly states: "These are **not**
  labeled pre-dock/post-dock ELITA captures and are not necessarily the same
  imaging modality"

### Classification

| Image | Pair Status |
|-------|-------------|
| eye_01 through eye_14 | **UNPAIRED** — independent images, no pairing information |

**No pairs exist for algorithmic validation.**

---

## 5. Image-Quality Analysis

### MEASURED (from Phase II report and code inspection)

The clinical proxy images have these characteristics:

| Property | Measured Range |
|----------|---------------|
| Resolution | 698x655 to 1600x1600 |
| Brightness (mean gray) | 24-116 (many dark) |
| Contrast (gray std) | 25-69 |
| Iris-region mean gray | 17-70 (dark iris under surgical light) |
| Iris texture (Laplacian abs-mean) | 1.2-3.3 (weak) |
| Near-white fraction in annulus | Mostly 0, 3 images ~0.9-1.1% |
| Format | JPEG (compression artifacts expected) |

### INFERENCE

The available proxy images are dark surgical views with weak iris texture.
This is consistent with surgical microscope illumination. Whether real ELITA
images will have stronger or weaker texture is unknown.

---

## 6. Ground-Truth Availability

### FACT

**No ground-truth rotation exists.** Specifically:

1. No manually measured cyclotorsion angles
2. No surgeon annotations of rotation
3. No device-provided orientation data
4. No image orientation markers
5. No toric axis reference data
6. No limbal/corneal reference markers
7. No manually annotated corresponding iris points
8. No pre/post rotational measurements

The `angle_deg` field in annotations is the **ellipse tilt angle** (geometric
orientation of the fitted ellipse), NOT a rotational ground truth for
cyclotorsion.

### Classification

**NO_GROUND_TRUTH** — no rotation reference of any kind exists.

### Validation Strategy When No Ground Truth Exists

When real ELITA pairs become available without rotation ground truth:

1. **Visual inspection**: overlay detected features on both images, manually
   assess whether correspondences appear correct
2. **Internal consistency**: check that multiple independent feature matches
   agree on the estimated rotation (low circular_std)
3. **Perturbation stability**: apply controlled rotations to one image and
   verify the estimated rotation shifts by the expected amount
4. **Cross-validation**: if multiple pairs exist from the same eye, verify
   consistent rotation estimates
5. **Manual annotation**: have a human annotator identify 5-10 corresponding
   iris landmarks and measure the rotation manually as a reference

None of these are equivalent to ground truth. They are consistency checks.

---

## 7. Iris ROI Feasibility

### FACT (from Phase II/III audit)

The iris ROI depends on the production `UnifiedDetector` producing both pupil
and limbus ellipses. The Phase II audit found:

| Image | Pupil Detected | Limbus Detected | Valid Iris ROI |
|-------|---------------|----------------|---------------|
| eye_01 | Yes | Yes | Yes |
| eye_02 | Yes | Yes | Yes |
| eye_03 | Yes | Yes | Yes |
| eye_06 | Yes | **No** | No |
| eye_07 | Yes | **No** | No |
| eye_08 | Yes | **No** | No |
| eye_09 | Yes | **No** | No |
| eye_10 | Yes | **No** | No |
| eye_11 | Yes | Yes | Yes |
| eye_12 | Yes | **No** | No |
| eye_13 | Yes | Yes | Yes |
| eye_14 | Yes | **No** | No |

**Valid iris ROI rate: 5/12 (42%)** on the clinical proxy set.

### INFERENCE

This is an **upstream bottleneck**, not an iris module defect. The iris layer
consumes pupil/limbus geometry; if the production detector cannot find the
limbus, the iris layer cannot operate.

For real ELITA images, the limbus detection rate may differ. If ELITA images
have different illumination, contrast, or field of view, the production
detector may perform better or worse.

### NOTE

The Phase III audit also noted that `masks['iris']` from the ONNX segmentation
engine could provide an alternative limbus source when the fitted ellipse is
absent. This was identified as a feasibility probe but not implemented.

---

## 8. Feature Detector Readiness

### TEST-VERIFIED

On synthetic test fixtures (320x320 artificial iris):
- Feature extraction is deterministic
- Quality filtering reduces flat-iris features
- Angular suppression enforces minimum separation
- Phase III hardening eliminates noise-induced spurious features
- Reflection mask correctly excludes specular regions

### MEASURED (Phase II/III, clinical proxy)

On the 5 valid proxy images (eye_01, 02, 03, 11, 13):

| Image | Accepted Features | Angular Coverage | Median Confidence |
|-------|------------------|-----------------|-------------------|
| eye_01 | 72 (capped) | 100% | 0.900 |
| eye_02 | 26 | 75% | 0.900 |
| eye_03 | 22 | 67% | 0.961 |
| eye_11 | 25 | 83% | 0.575 |
| eye_13 | 20 | 42% | 0.900 |

### INFERENCE

- Eye_01 has robust feature detection (72 features, full coverage)
- Eye_02/03 have moderate feature sets (22-26 features)
- Eye_11/13 have marginal feature sets (20-25 features, variable coverage)
- The feature detector works on dark surgical imagery but produces modest
  feature counts consistent with the low measured iris texture
- Whether real ELITA images will yield more or fewer features is unknown

### NOT TESTED

- Feature detector on real ELITA images (none available)
- Feature detector on images with different illumination characteristics
- Feature detector on images with different iris pigmentation
- Feature detector on images with significant eyelid occlusion
- Feature detector on images with surgical instruments in the field

---

## 9. Phase IV Correspondence Readiness

### TEST-VERIFIED (synthetic fixtures)

| Condition | Result |
|-----------|--------|
| Identity (0 deg) | Error <= 0.5 deg |
| +1 deg | Error <= 0.5 deg |
| -1 deg | Error <= 0.5 deg |
| +3 deg | Error <= 0.5 deg |
| -3 deg | Error <= 0.5 deg |
| +5 deg (lattice multiple) | Error <= 0.5 deg |
| 359 deg (wraparound) | Error <= 0.5 deg |
| Scale 1.05, rotation 0 | Rotation error <= 0.5 deg, scale error < 2% |
| Scale 1.03, rotation 3 | Rotation error <= 0.5 deg, scale error < 2% |
| Content mismatch | Correctly rejected |
| Translation-only | Correctly rejected |
| Too few matches | DEGENERATE (honest refusal) |
| Dense ambiguous | AMBIGUOUS (honest refusal) |
| Low similarity | LOW_SIMILARITY (honest refusal) |
| One-to-one matching | Verified deterministic |

**These results are on a single 320x320 synthetic fixture, not on the full
clinical proxy set.**

### NOT EXECUTED

The evaluation harness (`scripts/iris_phase4_correspondence_eval.py`) was
designed to run the full benchmark across 5 proxy images x ~15 conditions
(~75 cases). It has NOT been executed because:
1. It requires the production ONNX model (gitignored)
2. It requires clinical proxy images (gitignored in some environments)
3. The environment where this plan was created lacks those files

### INFERENCE

The correspondence architecture is sound on synthetic fixtures. The coarse
cyclic lattice search + NCC refinement approach works for controlled
transformations. Whether it will work on real image pairs with unknown
transformations, varying illumination, and different texture characteristics
is an open question that can only be answered with real data.

---

## 10. Synthetic-vs-Real Gap

### What Phase IV assumed (synthetic pairs)

| Assumption | Synthetic Reality | Real ELITA Reality |
|------------|------------------|-------------------|
| Texture preserved across rotation | Yes — same pixels, just rotated | **Unknown** — illumination may change |
| Illumination identical | Yes — same image, just warped | **Unknown** — pre/post may differ |
| Reflection behavior consistent | Synthetic disc added to B only | **Unknown** — real reflections are complex |
| No non-rigid deformation | Affine warp only | **Unknown** — eye may deform |
| Pupil size constant | Yes — same pupil in A and B | **Unknown** — dilation may differ |
| Full iris visible | Yes — same FOV | **Unknown** — cropping/occlusion may differ |
| No perspective change | 2D rotation only | **Unknown** — camera angle may shift |
| Feature visibility stable | Same content, just shifted | **Unknown** — lighting may hide features |

### INFERENCE

The synthetic benchmark validates **algorithmic correctness under idealized
conditions**. It does NOT validate robustness to the real-world variations
that ELITA pre/post-dock pairs will exhibit. This is a fundamental gap that
can only be closed with real paired data.

### Risks (not yet quantified)

1. **Illumination change**: pre-dock (sitting, slit lamp) vs post-dock
   (supine, surgical microscope) may have fundamentally different lighting
2. **Reflection pattern change**: different light sources create different
   specular patterns
3. **Pupil dilation**: pharmacological dilation or bright-light constriction
   may shift iris features radially
4. **Eyelid occlusion**: different eyelid positions may hide different parts
   of the iris
5. **Image quality**: different cameras, different compression, different
   resolution
6. **Perspective**: different camera angles may introduce non-rigid变形

---

## 11. Dataset Requirements

### ENGINEERING RECOMMENDATION (not clinical/statistical validation)

For a meaningful Phase V engineering pilot:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Unique eyes | 3 | 5-10 |
| Pre-dock images per eye | 1 | 1-2 |
| Post-dock images per eye | 1 | 1-2 |
| Valid pre/post pairs | 3 | 5-10 |
| Laterality available | Yes | Yes |
| Pairing metadata | Yes (which pre goes with which post) | Yes |
| Acquisition order known | Yes | Yes |
| Ground-truth rotation | Not required (see validation strategy) | Helpful if available |
| Image quality | Usable iris visibility | Usable iris visibility |

### What is NOT required for engineering validation

- Hundreds of eyes (that is clinical/statistical validation)
- Manual iris landmark annotations (helpful but not required)
- Multiple images per state (one pre + one post per eye is sufficient)
- Ground-truth rotation (consistency checks can substitute)

---

## 12. Annotation Requirements

### For engineering validation

| Annotation | Required? | Notes |
|------------|-----------|-------|
| Image pairing | **Yes** | Which pre-dock image goes with which post-dock image |
| Eye laterality | **Yes** | Left/right — prevents pairing left eye pre with right eye post |
| Pupil/limbus ellipses | **Yes** | Already produced by UnifiedDetector — verify on real ELITA images |
| Rotation ground truth | **No** (helpful) | Can use consistency checks instead |
| Iris landmark annotations | **No** (helpful) | Would enable direct correspondence validation |
| Quality grading | **No** | Can be derived from detection pipeline |

### For clinical validation (future, not Phase V)

- Large multi-eye dataset
- Multiple imaging sessions
- Reference axis annotations
- Surgeon-measured cyclotorsion
- Statistical significance requirements

---

## 13. Validation Protocol

### Phase V Validation Protocol (when real ELITA data is available)

#### Step 1: Data Ingestion
1. Receive paired pre-dock/post-dock ELITA images
2. Verify pairing (same eye, correct ordering)
3. Verify laterality consistency
4. Assess image quality (resolution, iris visibility, occlusion)

#### Step 2: Iris ROI Validation
1. Run `UnifiedDetector` on each image
2. Record pupil detection rate, limbus detection rate
3. If limbus detection fails, note as blocker
4. Build iris ROI from detected geometry
5. Record usable fraction, reflection coverage

#### Step 3: Feature Detection Evaluation
1. Run `detect_iris_features` on each image
2. Record feature count, angular/radial coverage
3. Assess spatial distribution (not clustered)
4. Compare feature counts between pre and post images
5. Visualize features on both images

#### Step 4: Correspondence Evaluation
1. Run `estimate_correspondence` on each pre/post pair
2. Record match count, failure classification
3. If OK: record estimated rotation, NCC scores, consensus fraction
4. If NOT OK: record failure reason and investigate

#### Step 5: Rotation Estimation Assessment
1. For successful pairs: check circular_std (internal consistency)
2. Compare rotation estimates across multiple pairs of the same eye (if available)
3. Visualize: overlay features from pre-dock onto post-dock using estimated rotation
4. Manual review: does the overlay appear correct?

#### Step 6: Failure Analysis
1. Classify all failures: DEGENERATE, LOW_NCC, LOW_SIMILARITY, HIGH_RESIDUAL, AMBIGUOUS
2. Investigate root causes
3. Determine which failures are data-quality issues vs algorithm limitations

#### Step 7: Reporting
1. Report per-image results honestly
2. Report aggregate statistics
3. Classify as: READY / NEEDS IMPROVEMENT / BLOCKED
4. Identify specific improvements needed

---

## 14. Acceptance Criteria

### FEATURE DETECTION SUCCESS

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| Pupil detection rate | >= 80% on real ELITA images | Production detector capability |
| Limbus detection rate | >= 50% on real ELITA images | Known bottleneck on proxy |
| Valid iris ROI rate | >= 50% of detected pairs | Minimum for correspondence |
| Feature count per image | >= 10 features | Minimum for correspondence |
| Angular coverage | >= 4 quadrants on at least 1 image per pair | Minimum for rotation estimation |

### CORRESPONDENCE SUCCESS

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| Match count per pair | >= 4 matches | `min_matches` in CorrespondenceConfig |
| Correspondence success rate | >= 50% of valid pairs | Engineering minimum |
| Failure honest refusal | 100% of failures are genuine (not false OK) | Correctness requirement |

### ROTATION ESTIMATION SUCCESS

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| Internal consistency | circular_std < 5.0 deg for OK pairs | Multiple features must agree |
| Visual overlay correctness | Manual review confirms overlay appears correct | No ground truth available |
| Cross-pair consistency | If multiple pairs exist, rotation estimates agree within 10 deg | Consistency check |

### DATA QUALITY SUCCESS

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| Pair validity | All tested pairs are verified same-eye | Prevents false validation |
| Laterality consistency | Left pre matches left post | Prevents impossible pairs |
| Iris visibility | >= 50% of iris annulus visible | Minimum for feature extraction |

---

## 15. Risk Register

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| 1 | No real ELITA data exists | **BLOCKER** | CERTAIN | Data must be acquired externally |
| 2 | Synthetic benchmark not executed | HIGH | CERTAIN | Run `iris_phase4_correspondence_eval.py` when model + data available |
| 3 | Limbus detection fails on real ELITA | HIGH | MEDIUM | Alternative ROI from ML iris mask; or improve limbus detector separately |
| 4 | Real iris texture too weak for features | HIGH | MEDIUM | Evaluate honestly; may need learned descriptors |
| 5 | Illumination changes destroy correspondence | HIGH | MEDIUM | Cannot mitigate without real data |
| 6 | Pupil dilation shifts features radially | MEDIUM | HIGH | Iris-relative coordinates should handle this; verify on real data |
| 7 | No rotation ground truth | MEDIUM | CERTAIN | Use consistency checks + manual review |
| 8 | Feature detector confidence compressed | LOW | CERTAIN | Renormalize; low priority |
| 9 | 5-degree lattice ceiling limits precision | LOW | CERTANT | NCC refinement addresses this; verified in tests |

---

## 16. Recommended Next Implementation Phase

### Decision: F. SYNTHETIC BENCHMARK MUST BE COMPLETED FIRST

**Rationale:**

Before any real ELITA validation can be meaningful, the synthetic benchmark
must be executed to establish the **ceiling of current capability** under
controlled conditions. Without this baseline:

- We cannot distinguish "algorithm limitation" from "real-world degradation"
- We cannot set expectations for real-data performance
- We cannot identify which failure modes are expected vs surprising

### Recommended Path

**Phase V-A: Complete Synthetic Benchmark** (next implementation)

1. Run `scripts/iris_phase4_correspondence_eval.py` on the clinical proxy set
2. Report full results across all transformation conditions
3. Establish per-image and aggregate performance baselines
4. Identify which images/conditions are strongest/weakest
5. Commit results as `IRIS_PHASE4_BENCHMARK_RESULTS.md`

**Phase V-B: Real ELITA Data Acquisition** (external dependency)

6. Acquire real paired pre-dock/post-dock ELITA images
7. Ingest into `clinical_data/elita/` (or similar)
8. Verify pairing, laterality, image quality
9. This step is **outside the scope of code implementation** — it requires
   clinical collaboration and data sharing agreements

**Phase V-C: Real ELITA Validation** (after data is available)

10. Run the same evaluation harness on real ELITA pairs
11. Compare results to synthetic benchmark baseline
12. Investigate failures
13. Determine readiness for further development

### What NOT to do next

- Do NOT implement clinical cyclotorsion estimation
- Do NOT implement astigmatism correction
- Do NOT integrate into the GUI
- Do NOT modify production detection
- Do NOT train learned models
- Do NOT make clinical claims

---

## 17. Proposed Architecture

### Future production pipeline (conceptual, not implemented)

```
ELITA IMAGE (pre-dock or post-dock)
    |
    +-- Image quality assessment
    |
    +-- Pupil / limbus geometry (UnifiedDetector)
    |       |
    |       +-- If limbus missing: alternative ROI from ML iris mask
    |
    +-- Iris ROI construction (IrisROIExtractor)
    |
    +-- Occlusion / reflection mask (IrisMasking)
    |
    +-- Iris feature detection (IrisFeatureExtractor)
    |       |
    |       +-- Phase III hardened response
    |       +-- Quality gating
    |       +-- Angular suppression
    |
    +-- Feature quality assessment
    |
    +-- Feature representation (IrisFeatureSet)
            |
            +-- angle_deg, radial_norm (iris-relative)
            +-- descriptor (16-bin histogram)
            +-- confidence
            +-- visibility
            |
            v
      PRE/POST PAIR
            |
            v
    Feature correspondence (estimate_correspondence)
            |
            +-- Coarse cyclic lattice search
            +-- Descriptor similarity
            +-- Sub-lattice NCC refinement
            |
            v
    Robust geometric estimation
            |
            +-- Consensus estimator (default)
            +-- RANSAC (alternative)
            |
            v
    Rotation / cyclotorsion angle
            |
            v
    Clinical orientation output
            |
            v
    Future astigmatism-axis support
```

### Component Status

| Component | Status |
|-----------|--------|
| Image quality assessment | **MISSING** — not implemented |
| Pupil/limbus geometry | **EXISTS** — UnifiedDetector (production) |
| Alternative limbus ROI | **MISSING** — identified but not implemented |
| Iris ROI construction | **EXISTS** — IrisROIExtractor |
| Occlusion/reflection mask | **EXISTS** — IrisMasking |
| Iris feature detection | **EXISTS** — Phase III hardened |
| Feature quality assessment | **PARTIALLY EXISTS** — confidence score, but compressed |
| Feature representation | **EXISTS** — IrisFeatureSet |
| Feature correspondence | **EXISTS** — Phase IV prototype (evaluation only) |
| Robust geometric estimation | **EXISTS** — consensus/RANSAC estimators |
| Rotation estimation | **EXISTS** — but not validated on real data |
| Clinical orientation output | **MISSING** — not implemented |
| Astigmatism-axis support | **MISSING** — future phase |

---

## 18. Explicit Non-Goals

Phase V does NOT:

1. Implement clinical cyclotorsion estimation
2. Implement astigmatism correction
3. Make clinical claims about accuracy
4. Integrate into the GUI
5. Modify production detection pipeline
6. Modify calibration
7. Modify pupil detection
8. Modify limbus detection
9. Train learned models
10. Add new dependencies
11. Create production functionality
12. Replace the need for real ELITA validation

---

## 19. Final Decision

### Classification: DATA ACQUISITION REQUIRED

The current technical implementation is **sound at the synthetic level**:
- 59/59 iris tests pass
- Feature detection works on clinical proxy images
- Correspondence recovers synthetic rotations to sub-0.5 deg accuracy
- Failure modes are correctly detected and honestly refused

However, **no real ELITA data exists** in the repository. Without real paired
pre-dock/post-dock images:

- The synthetic benchmark ceiling is unknown (harness not executed)
- Real-world correspondence performance is unknown
- Rotation estimation accuracy on real data is unknown
- Clinical feasibility cannot be assessed

### Recommended Action

1. **IMMEDIATE**: Execute the synthetic benchmark (`iris_phase4_correspondence_eval.py`)
   to establish the performance ceiling
2. **DEPENDENT**: Acquire real ELITA paired images (external dependency)
3. **AFTER DATA**: Run identical validation harness on real ELITA pairs
4. **DECISION POINT**: If real performance matches synthetic ceiling -> proceed
   to clinical pilot. If not -> investigate and improve.

### Blocker

**The single blocker is the absence of real ELITA paired data.** All other
work (synthetic benchmark, alternative ROI, confidence renormalization) can
proceed in parallel but does not substitute for real data validation.

---

## 20. Validation Level Summary

| Level | Status |
|-------|--------|
| Synthetic implementation correctness | **VERIFIED** (59 tests) |
| Complete synthetic benchmark | **NOT EXECUTED** (harness exists, needs model + data) |
| Real ELITA validation | **BLOCKED** (no data) |
| Clinical validation | **NOT IN SCOPE** |

---

## 21. File Inventory

### Existing Phase IV files (committed)

| File | Commit | Lines |
|------|--------|-------|
| `pupil_tracking/iris/paired.py` | `c39e5b5` | 199 |
| `pupil_tracking/iris/correspondence.py` | `eeba8c2` | 994 |
| `pupil_tracking/tests/test_iris_paired.py` | `1726a1c` | 201 |
| `pupil_tracking/tests/test_iris_correspondence.py` | `65ccbe8` | 352 |
| `IRIS_PHASE4_REPORT.md` | `f6466b9` | 679 |

### Existing but untracked

| File | Purpose |
|------|---------|
| `scripts/iris_phase4_correspondence_eval.py` | Evaluation harness (requires model + data) |

### New in this plan

| File | Purpose |
|------|---------|
| `IRIS_PHASE5_PLAN.md` | This document |

---

*This is a planning document. No code was modified. No tests were run beyond
verifying the current state. No implementation was performed.*

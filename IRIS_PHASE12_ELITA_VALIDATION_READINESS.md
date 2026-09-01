# IRIS PHASE XII — ELITA PRE/POST-DOCK VALIDATION READINESS

**Date:** 2026-09-01
**HEAD:** `6fd0436`
**Scope:** Validation-readiness audit. No algorithm changes. No clinical claims.

---

## 1. Purpose

Determine exactly what is required to validate the current iris and cyclotorsion pipeline on real ELITA pre-dock/post-dock images. This is a readiness assessment, not an implementation phase.

---

## 2. Current Iris/Cyclotorsion Capability

### What Is Implemented

**FACT:** The following modules are implemented and tested (103/103 tests pass):

| Module | File | Purpose |
|--------|------|---------|
| Types | `types.py` | IrisROI, IrisFeature, IrisFeatureSet, IrisDetectionResult |
| Config | `config.py` | IrisConfig (ROI insets, lattice, quality thresholds) |
| ROI | `roi.py` | Annular ROI from pupil/limbus ellipses |
| Masking | `masking.py` | Reflection/occlusion masking, usable mask construction |
| Normalization | `normalization.py` | Iris-relative coordinate mapping |
| Extraction | `extraction.py` | 72-angle × 8-radius feature lattice extraction |
| Detection | `detect.py` | Top-level IrisFeatureDetector orchestration |
| Correspondence | `correspondence.py` | Matching, NCC refinement, rotation estimation |
| Paired | `paired.py` | Synthetic pair generation for benchmark |
| Robustness | `robustness.py` | Repeatability, spatial distribution, quality stability |

### What Is NOT Implemented

**FACT:** The following are explicitly out of scope:

- Real ELITA data ingestion
- Clinical validation
- Pentacam detection
- Cross-system registration
- Production integration of iris
- Learned matching models
- Real paired-image evaluation

### Current Pipeline Contract

```
Input:
  image (BGR or grayscale)
  + pupil EllipseParams
  + limbus EllipseParams

    ↓

ROI Construction:
  Annular ring between pupil and limbus ellipses
  Insets: 12% inner, 12% outer (configurable)

    ↓

Masking:
  Reflection removal
  Occlusion handling
  Usable mask construction

    ↓

Feature Extraction:
  72-angle × 8-radius lattice (5° angular step)
  Intensity histogram descriptors (16-bin)
  Confidence scoring
  Angular suppression (min 5° separation)

    ↓

Correspondence (pair A → B):
  Coarse cyclic lattice matching (5° steps)
  NCC sub-lattice refinement (±2.5°)
  Per-pair rotation estimates

    ↓

Rotation Estimation:
  Consensus (modal binning)
  Global spatial consistency (circular voting)
  Global hybrid (GC when confident, consensus fallback)
  RANSAC (exhaustive two-point)

    ↓

Evidence/Confidence:
  Feature count, angular coverage, largest gap
  Global inlier count/frac/std
  LOW_EVIDENCE gate (configurable, disabled by default)
  Failure classification (DEGENERATE/LOW_NCC/LOW_SIMILARITY/
    HIGH_RESIDUAL/AMBIGUOUS/LOW_EVIDENCE/OK)

    ↓

Output:
  CorrespondenceResult:
    estimated_rotation_deg
    estimated_scale
    failure / failure_reason
    valid (bool)
    feature_count, angular_coverage_ratio, ...
    global_inlier_count, global_inlier_frac, ...
    processing_time_ms
```

---

## 3. Manager-Directed Project Sequence

**FACT:** The intended project flow is:

```
ELITA pre-dock
      +
ELITA post-dock
      ↓
Iris feature detection
      ↓
Iris feature matching
      ↓
Rotation/cyclotorsion angle
      ↓
Confidence in ELITA result
      ↓
Pentacam sitting
      +
ELITA supine
      ↓
Cross-system registration
      ↓
Sitting-to-supine rotational relationship
```

**Phase XII scope:** Prepare the ELITA stage (steps 1–5) for real-data validation.

**NOT in scope:** Pentacam detection, cross-system registration, production integration.

---

## 4. Current Limitations

**FACT (from benchmark evaluation on 5 clinical proxy images):**

| Limitation | Evidence |
|-----------|----------|
| Sparse features (eye_13: 9 features, 45° span) | 3 FALSE-OK cases in benchmark |
| Feature count alone insufficient for reliability | angular_coverage_ratio is 5.6× stronger predictor |
| Synthetic-only validation | Real ELITA pairs not available |
| 5-image benchmark | May not generalize |
| No real paired data | Cannot validate on clinical reality |
| Eye_11 degenerate (3 features) | All 6 cases FAILED |
| Search window ±7.5° | Rotations > 7.5° unreachable |

**INFERENCE:** The synthetic benchmark demonstrates the algorithm works on controlled transformations of real iris images. Real ELITA validation is the necessary next step.

---

## 5. Real ELITA Data Requirements

### 5.1 Per-Image Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Pre-dock image** | REQUIRED | Image captured before surgery (sitting position) |
| **Post-dock image** | REQUIRED | Image captured after surgery (supine position) |
| **Same eye** | REQUIRED | Both images must be of the same eye |
| **Eye laterality** | REQUIRED | Left/right identification |
| **Image format** | REQUIRED | JPEG, PNG, or raw; must be readable by OpenCV |
| **Pupil/limbus geometry** | REQUIRED | Must be detectable by existing UnifiedDetector |
| **Iris visible** | REQUIRED | Iris must not be fully occluded |
| **Image resolution** | PREFERRED | Sufficient for pupil/limbus detection (≥320×240) |
| **Metadata** | PREFERRED | Patient ID, capture time, device info |
| **Timestamps** | OPTIONAL | Pre/post timing |
| **Image quality score** | OPTIONAL | Sharpness, contrast metrics |
| **Ground-truth rotation** | UNKNOWN | May not exist; see §7 |

### 5.2 Pairing Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Pre-dock ↔ post-dock pairing** | REQUIRED | Explicit mapping of which images form a pair |
| **Anonymized patient ID** | REQUIRED | For grouping pairs by patient |
| **Eye laterality per pair** | REQUIRED | Left/right per pair |
| **Sitting/supine labels** | REQUIRED | Which image is sitting, which is supine |

### 5.3 What Is NOT Required (Yet)

- Pentacam images
- Cross-system registration
- Clinical ground truth thresholds
- Production deployment
- Real-time processing

---

## 6. Ground-Truth / Reference Rotation

**FACT:** No source of ground-truth cyclotorsion rotation is currently available.

### Possible Reference Sources

| Source | Feasibility | Notes |
|--------|-------------|-------|
| Manual annotation | UNKNOWN | Requires expert marking of corresponding iris landmarks |
| Device-provided rotation | UNKNOWN | Some devices may record head/eye orientation |
| Known phantom rotation | UNKNOWN | Controlled experiment with known rotation |
| Consensus across pairs | UNKNOWN | Multiple independent measurements |
| No ground truth | CURRENT STATE | Cannot compute absolute error |

### Validation Without Ground Truth

If no reference rotation is available, validation must proceed by:

1. **Internal consistency** — Does the algorithm produce consistent results across multiple runs on the same pair?
2. **Determinism** — Are results reproducible given identical inputs?
3. **Quality metrics** — Do feature count, angular coverage, and inlier fraction indicate reliable estimation?
4. **Honest rejection** — Does the algorithm correctly identify insufficient-evidence cases?
5. **Bounded error** — Is the estimated rotation within a plausible physiological range (0–35° cyclotorsion)?

**INFERENCE:** Internal consistency and honest rejection can be validated without ground truth. Absolute accuracy requires a reference rotation source.

---

## 7. Validation Metrics

### 7.1 Algorithmic Metrics (No Ground Truth Required)

| Metric | Definition | Source |
|--------|-----------|--------|
| Feature count | Number of accepted features | `CorrespondenceResult.feature_count` |
| Angular coverage | Smallest arc containing all features / 360 | `CorrespondenceResult.angular_coverage_ratio` |
| Largest gap | Largest angular gap between features | `CorrespondenceResult.largest_angular_gap` |
| Global inlier count | Number of estimates agreeing with dominant peak | `CorrespondenceResult.global_inlier_count` |
| Global inlier fraction | Inlier count / total estimates | `CorrespondenceResult.global_inlier_frac` |
| Failure kind | Why estimation succeeded or failed | `CorrespondenceResult.failure` |
| Valid | Whether a rotation was emitted | `CorrespondenceResult.valid` |
| Processing time | End-to-end runtime | `CorrespondenceResult.processing_time_ms` |

### 7.2 Reference-Dependent Metrics (Ground Truth Required)

| Metric | Definition | Status |
|--------|-----------|--------|
| Circular angular error | min|θ̂ - θ_ref| mod 360 | BLOCKED (no reference) |
| Absolute angular error | |θ̂ - θ_ref| | BLOCKED |
| Median/mean/max error | Distribution of errors | BLOCKED |
| Rejection rate | Fraction of pairs rejected | Available |
| Insufficient-evidence rate | Fraction flagged LOW_EVIDENCE | Available |
| Repeatability | Consistency across repeated captures | UNKNOWN (requires repeat data) |

---

## 8. Quality / Evidence Gates

### 8.1 Currently Implemented Gates

| Gate | Config Parameter | Threshold | Effect |
|------|-----------------|-----------|--------|
| Min matches | `min_matches` | 4 | DEGENERATE if < 4 matches |
| NCC gate | `ncc_min` | 0.42 | Low NCC ratio check |
| Consensus fraction | `min_consensus_fraction` | 0.5 | HIGH_RESIDUAL if low |
| Inlier std | `residual_std_max_deg` | 2.0 | HIGH_RESIDUAL if high |
| Ambiguity | `ambiguity_ratio_max` | 0.5 | AMBIGUOUS if high |
| Descriptor similarity | `low_similarity_ratio_max` | 0.5 | LOW_SIMILARITY if low |
| Evidence gate | `evidence_gate` | False (off) | LOW_EVIDENCE if sparse |

### 8.2 Recommended for Real ELITA Validation

| Gate | Recommendation | Rationale |
|------|---------------|-----------|
| evidence_gate | **Enable** | Eliminates FALSE-OK on sparse cases |
| evidence_min_features | 4 | Current default |
| evidence_min_angular_coverage | 0.20 | Current default |
| evidence_min_occupied_bins | 3 | Current default |
| rotation_method | `"global_hybrid"` | Best balance (Phase X) |

### 8.3 Gates Requiring Real-Data Calibration

| Gate | Current Value | Real-Data Status |
|------|--------------|-----------------|
| ncc_min | 0.42 | Tuned on synthetic; may need adjustment |
| min_matches | 4 | Conservative; may need lowering for sparse real data |
| residual_std_max_deg | 2.0 | Tuned on synthetic |
| evidence thresholds | 0.20, 4, 3 | May need real-data validation |

---

## 9. Proposed Validation Dataset Structure

### 9.1 Directory Layout (Proposed, Not Implemented)

```
elita_validation/
  manifest.json              # pairing and metadata
  pre_dock/
    patient_001_L.jpeg
    patient_001_R.jpeg
    patient_002_L.jpeg
    ...
  post_dock/
    patient_001_L.jpeg
    patient_001_R.jpeg
    patient_002_L.jpeg
    ...
```

### 9.2 Manifest Schema (Proposed)

```json
{
  "version": "1.0",
  "pairs": [
    {
      "pair_id": "p001_L",
      "patient_id": "anonymized_001",
      "eye": "left",
      "pre_dock": "pre_dock/patient_001_L.jpeg",
      "post_dock": "post_dock/patient_001_L.jpeg",
      "sitting_image": "pre_dock/patient_001_L.jpeg",
      "supine_image": "post_dock/patient_001_L.jpeg",
      "reference_rotation_deg": null,
      "notes": ""
    }
  ]
}
```

### 9.3 What Should NOT Be in the Repository

- Real patient images (gitignored)
- Patient-identifiable information
- Unanonymized data
- Large binary files

---

## 10. Proposed Experiment Design

### 10.1 Experiment A: Basic Pre/Post Validation

1. Load each pre-dock image
2. Run `detect_iris_features()` → feature_set_a
3. Load paired post-dock image
4. Run `detect_iris_features()` → feature_set_b
5. Run `estimate_correspondence()` with `rotation_method="global_hybrid"`
6. Enable `evidence_gate=True`
7. Record: rotation estimate, failure kind, quality metrics

### 10.2 Experiment B: Feature Quality Assessment

For each image pair, report:
- Feature count (A and B)
- Angular coverage (A and B)
- Global inlier count/frac
- Failure kind
- Processing time

### 10.3 Experiment C: Sparse-Feature Analysis

Identify pairs with:
- angular_coverage < 0.30
- feature_count < 15
- global_inlier_count < 5

Report these as LOW_EVIDENCE candidates.

### 10.4 Experiment D: Consistency Check

If repeat captures are available:
- Run same pair multiple times
- Verify identical results (determinism)
- Check for sensitivity to image quality variations

### 10.5 Dataset Split

| Split | Purpose | Size |
|-------|---------|------|
| Development | Algorithm tuning | Small (2–3 pairs) |
| Validation | Evidence gate calibration | Medium (5–10 pairs) |
| Holdout | Final evaluation | Remaining |

**BLOCKED:** Without real data, splits cannot be defined.

---

## 11. Engineering Acceptance Criteria

| Criterion | Metric | Threshold | Status |
|-----------|--------|-----------|--------|
| Determinism | Repeated runs identical | 100% | VERIFIED (synthetic) |
| Valid output | `valid=True` or honest failure | Required | IMPLEMENTED |
| Bounded rotation | 0–360° | Always | IMPLEMENTED |
| Honest rejection | LOW_EVIDENCE for sparse cases | Required | IMPLEMENTED |
| No false confidence | FALSE-OK rate = 0 with gate | Required | VERIFIED (synthetic) |
| Runtime | < 400ms per pair | ~50ms | VERIFIED |
| No production changes | iris additive | Required | VERIFIED |
| Test suite | 103/103 pass | Required | VERIFIED |

---

## 12. Clinical Acceptance Criteria

**BLOCKED / TBD:** Clinical thresholds must be provided by the clinical team.

**UNKNOWN:** What angular error is clinically acceptable for cyclotorsion measurement?

Potential clinical questions:
- Is ≤ 1° error acceptable?
- Is ≤ 2° error acceptable?
- Is ≤ 5° error acceptable?
- What rejection rate is clinically acceptable?
- What image quality is required?

**RECOMMENDATION:** Do not invent clinical thresholds. Wait for clinical team input.

---

## 13. Validation Harness Readiness

### 13.1 Current Tooling

| Tool | Status | Can Accept Real Data? |
|------|--------|----------------------|
| `estimate_correspondence()` | Implemented | YES — takes images + feature sets |
| `evaluate_pair()` | Implemented | YES — but requires ground truth |
| `detect_iris_features()` | Implemented | YES — takes image + geometry |
| `UnifiedDetector` | Implemented | YES — detects pupil/limbus |
| `compute_feature_metrics()` | Implemented | YES — computes coverage metrics |
| `scripts/iris_phase4_correspondence_eval.py` | Implemented | PARTIALLY — needs real data path |
| `scripts/iris_phase11_sparse_analysis.py` | Implemented | YES — compares methods |

### 13.2 What Needs to Change for Real ELITA Data

| Change | Scope | Risk |
|--------|-------|------|
| Add real data path to eval script | Validation tooling only | None |
| Add manifest parser | Validation tooling only | None |
| Add result CSV/JSON output | Validation tooling only | None |
| Add quality report generation | Validation tooling only | None |

**VERDICT:** No algorithm changes needed. Only validation tooling additions.

---

## 14. Pentacam Boundary

### What the Future Pentacam Stage Needs From ELITA

**FACT:** The Pentacam stage (future, not this phase) will need:

| Item | Description | Status |
|------|-------------|--------|
| ELITA rotation estimate | Cyclotorsion angle from pre-dock/post-dock | Available |
| ELITA confidence | Whether estimate is reliable | Available |
| ELITA feature set | Features from supine image | Available |
| Iris ROI | Geometry of detected iris | Available |
| Image quality metrics | Whether image is usable | Available |

### What the Pentacam Stage Does NOT Need From ELITA

- Pupil/limbus geometry (will re-detect from Pentacam image)
- Feature extraction (will re-extract from Pentacam image)
- Pre-dock image (Pentacam has its own sitting image)

### Interface Definition (Future)

```python
# Future Pentacam stage will consume:
{
    "rotation_estimate_deg": float,  # from ELITA
    "rotation_confidence": str,       # "OK" / "LOW_EVIDENCE" / "DEGENERATE"
    "supine_feature_set": IrisFeatureSet,  # features from supine image
    "supine_roi": IrisROI,            # ROI from supine image
    "image_quality": dict,            # quality metrics
}
```

**DO NOT IMPLEMENT NOW.** This is documentation of the future interface.

---

## 15. Current Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| No real ELITA pre-dock images | CANNOT validate | Obtain from clinical team |
| No real ELITA post-dock images | CANNOT validate | Obtain from clinical team |
| No pairing manifest | CANNOT run experiments | Create with clinical team |
| No ground-truth rotation | CANNOT compute absolute error | May not exist; use internal consistency |
| Clinical thresholds not defined | CANNOT define acceptance | Clinical team input required |

---

## 16. Readiness Checklist

| Item | Status | Notes |
|------|--------|-------|
| Real ELITA pre-dock images available | **BLOCKED** | Not in repository |
| Real ELITA post-dock images available | **BLOCKED** | Not in repository |
| Pairing available | **BLOCKED** | No manifest exists |
| Eye laterality available | **BLOCKED** | Depends on data |
| Sitting/supine labels available | **BLOCKED** | Depends on data |
| Ground-truth/reference rotation available | **UNKNOWN** | May not exist |
| Image quality acceptable | **UNKNOWN** | Depends on data |
| Iris ROI detected | **READY** | Implemented, tested |
| Iris features detected | **READY** | Implemented, tested |
| Sufficient angular coverage | **READY** | Metrics computed |
| Correspondence successful | **READY** | Implemented, tested |
| Rotation estimated | **READY** | Multiple methods available |
| Evidence/confidence available | **READY** | LOW_EVIDENCE gate implemented |
| Validation metrics calculated | **READY** | compute_feature_metrics available |
| Holdout evaluation possible | **BLOCKED** | No data to split |

---

## 17. Next Actions

| # | Action | Owner | Status |
|---|--------|-------|--------|
| 1 | Obtain real ELITA pre-dock images | Clinical team | BLOCKED |
| 2 | Obtain real ELITA post-dock images | Clinical team | BLOCKED |
| 3 | Create pairing manifest | Clinical team | BLOCKED |
| 4 | Define eye laterality format | Clinical + engineering | BLOCKED |
| 5 | Determine if ground truth exists | Clinical team | UNKNOWN |
| 6 | Define clinical error thresholds | Clinical team | TBD |
| 7 | Add real data path to eval script | Engineering | READY |
| 8 | Add manifest parser | Engineering | READY |
| 9 | Run first real-data experiment | Engineering | BLOCKED on 1–3 |
| 10 | Calibrate evidence gates on real data | Engineering | BLOCKED on 1–3 |

---

## 18. Final Recommendation

**VERDICT: READY FOR REAL-DATA VALIDATION (blocked on data availability).**

The iris pipeline is technically ready:
- All modules implemented and tested (103/103 tests)
- Multiple rotation methods available
- Evidence gate implemented (configurable)
- Sparse-feature metrics available
- Validation tooling exists and can accept real data
- Production safety verified

The only blocker is **real ELITA data**. Once pre-dock/post-dock images and a pairing manifest are available, the first real-data experiment can be run with minimal tooling changes.

**DO NOT:**
- Start Pentacam implementation
- Change algorithm without evidence
- Claim clinical accuracy
- Fabricate ELITA data
- Integrate iris into production

**DO:**
- Wait for real ELITA data
- Run first experiment when data arrives
- Calibrate evidence gates on real data
- Document real-data findings

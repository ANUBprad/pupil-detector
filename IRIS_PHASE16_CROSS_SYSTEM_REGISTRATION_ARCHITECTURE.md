# IRIS PHASE XVI — CROSS-SYSTEM REGISTRATION ARCHITECTURE & FINAL INTEGRATION READINESS

**Date:** 2026-09-01
**HEAD:** `0567a37`
**Scope:** Cross-system registration architecture, contracts, and final integration readiness. No real data available.

---

## 1. Objective

Create the final architecture and contract for Pentacam ↔ ELITA cross-system correspondence and transformation. Define what each system provides, what the shared representation is, what coordinate systems exist, what transformations are required, and what data is needed.

---

## 2. Current Project Pipeline

```
ELITA PRE-DOCK (SITTING)
        ↓
Iris feature detection
        ↓
ELITA POST-DOCK (SUPINE)
        ↓
Iris feature matching
        ↓
ELITA cyclotorsion estimate
        ↓
Confidence / evidence
        ↓
PENTACAM SITTING IMAGE          ← CURRENT PHASE
        ↓
Pentacam feature detection
        ↓
PENTACAM ↔ ELITA correspondence
        ↓
Cross-system transformation
        ↓
Sitting → Supine rotational relationship
```

---

## 3. ELITA Capability

**FACT:** The ELITA iris pipeline provides:

| Output | Type | Source |
|--------|------|--------|
| Iris features | `IrisFeatureSet` | `detect_iris_features()` |
| Feature coordinates | `IrisFeature.x, y` | Image pixel space |
| Feature iris-relative coords | `IrisFeature.angle_deg, radial_norm` | Normalized to ROI |
| Feature descriptors | `IrisFeature.descriptor` | 16-bin intensity histogram |
| Pupil geometry | `EllipseParams` | `UnifiedDetector` |
| Limbus geometry | `EllipseParams` | `UnifiedDetector` |
| ROI geometry | `IrisROI` | Annular ring |
| Cyclotorsion estimate | `CorrespondenceResult.estimated_rotation_deg` | Pre→post rotation |
| Rotation confidence | `CorrespondenceResult.failure` | OK/DEGENERATE/LOW_NCC/... |
| Evidence metrics | `feature_count, angular_coverage_ratio, ...` | Sparse metrics |

**Status:** DESIGNED + IMPLEMENTED + TESTED (103/103)

---

## 4. Pentacam Capability

**FACT:** The Pentacam module provides:

| Output | Type | Status |
|--------|------|--------|
| Detection result | `PentacamDetectionResult` | DESIGNED |
| Pupil geometry | `PentacamGeometry.pupil` (EllipseParams) | DESIGNED |
| Limbus geometry | `PentacamGeometry.limbus` (EllipseParams) | DESIGNED |
| Iris features | `PentacamFeatureSet` | DESIGNED |
| Feature coordinates | `PentacamFeature.x, y` | DESIGNED |
| Quality assessment | `PentacamQuality` | DESIGNED |
| Coordinate system label | `PentacamDetectionResult.coordinate_system` | DESIGNED |

**Status:** DESIGNED + SYNTHETIC TESTS ONLY. NOT VALIDATED on real data.

---

## 5. Current Data Availability

| Dataset | Status | Notes |
|---------|--------|-------|
| Real ELITA pre-dock/post-dock pairs | BLOCKED | Not in repository |
| Real Pentacam sitting images | BLOCKED | Not in repository |
| Pentacam ↔ ELITA paired data | BLOCKED | Not in repository |
| Clinical images (surgical eyes) | AVAILABLE | 12 images, not Pentacam |
| Synthetic fixtures | AVAILABLE | For software testing only |

**NO REAL CROSS-SYSTEM VALIDATION IS POSSIBLE YET.**

---

## 6. ELITA Output Contract

### Required for Registration

| Item | Required? | Source |
|------|-----------|--------|
| `IrisFeatureSet.features` | YES | Feature coordinates + descriptors |
| `IrisROI` | YES | Pupil/limbus geometry reference |
| `CorrespondenceResult.estimated_rotation_deg` | YES | ELITA cyclotorsion |
| `CorrespondenceResult.failure` | YES | Confidence/validity |
| `IrisFeature.x, y` | YES | Image-space coordinates |
| `IrisFeature.angle_deg, radial_norm` | USEFUL | Normalized coordinates |
| `IrisFeature.descriptor` | USEFUL | For cross-system matching |
| `IrisFeature.confidence` | USEFUL | Feature quality weighting |
| `CorrespondenceResult.global_inlier_frac` | USEFUL | Rotation reliability |
| `CorrespondenceResult.feature_count` | USEFUL | Evidence assessment |
| `CorrespondenceResult.angular_coverage_ratio` | USEFUL | Evidence assessment |

### Not Required

| Item | Reason |
|------|--------|
| Raw image | Not needed if features are sufficient |
| Mask statistics | Internal to ELITA |
| Processing time | Diagnostic only |

---

## 7. Pentacam Output Contract

### Designed (Not Validated)

| Item | Required? | Status |
|------|-----------|--------|
| `PentacamDetectionResult.valid` | YES | DESIGNED |
| `PentacamGeometry.pupil` | YES | DESIGNED |
| `PentacamGeometry.limbus` | YES | DESIGNED |
| `PentacamFeatureSet.features` | YES | DESIGNED |
| `PentacamFeature.x, y` | YES | DESIGNED |
| `PentacamFeature.angle_deg, radial_norm` | USEFUL | DESIGNED |
| `PentacamFeature.descriptor` | USEFUL | DESIGNED |
| `PentacamQuality` | YES | DESIGNED |
| `coordinate_system` | YES | DESIGNED |

---

## 8. Common Representation

### What Can Connect Pentacam Sitting with ELITA Supine?

| Candidate | Connection Value | Limitations |
|-----------|-----------------|-------------|
| **Pupil geometry** | HIGH — stable anatomical landmark | Different coordinate systems |
| **Limbus geometry** | HIGH — defines iris extent | Different coordinate systems |
| **Iris features** | HIGH — texture-based matching | Different illumination/geOMETRY |
| **Normalized coordinates** | MEDIUM — device-independent | Requires calibration |
| **Feature descriptors** | MEDIUM — if illumination compatible | Unknown compatibility |

### Recommended Common Representation

**Geometry-based + feature-based hybrid:**

1. **Primary:** Pupil and limbus ellipse parameters (center, radii, orientation)
   - Both systems detect these
   - Provides geometric anchor points
   - Device-independent anatomical reference

2. **Secondary:** Iris feature coordinates in normalized polar space
   - `angle_deg` (CCW from x-axis)
   - `radial_norm` (0=pupil, 1=limbus)
   - Device-independent if ROI normalization is consistent

3. **Tertiary:** Feature descriptors (if illumination allows matching)
   - May not be directly compatible between systems
   - Requires empirical validation

---

## 9. Coordinate-System Analysis

### ELITA Coordinate System (VERIFIED)

- **Origin:** Top-left of image
- **X-axis:** Positive rightward
- **Y-axis:** Positive downward
- **Angle convention:** Counter-clockwise from positive x-axis, [0, 180)
- **Scale:** Image pixels (device/resolution dependent)
- **Iris-relative:** `angle_deg` CCW from x-axis, `radial_norm` (0=pupil, 1=limbus)

### Pentacam Coordinate System (UNKNOWN)

- **Expected:** Device-specific pixel coordinates
- **Origin:** Likely device-specific (may not be top-left)
- **Axes:** Likely standard image axes but not guaranteed
- **Projection:** Scheimpflug cross-section differs from en-face
- **Scale:** Device/resolution dependent

### Transformations Required

| Transformation | Classification | Rationale |
|---------------|---------------|-----------|
| Translation | REQUIRED | Different devices, different positions |
| Rotation | REQUIRED | Different mounting/orientation |
| Uniform scale | POSSIBLE | Different magnification |
| Non-uniform scale | POSSIBLE | Different anamorphic ratio |
| Perspective | UNLIKELY | Similar viewing distance |
| Optical distortion | UNKNOWN | Different lens systems |
| Nonlinear distortion | UNKNOWN | Different projection models |

**CRITICAL:** The relationship must be determined from real paired data.

---

## 10. Transformation Model

### Recommended Progression

1. **Rigid 2D** (rotation + translation) — simplest, test first
2. **Similarity 2D** (rotation + translation + uniform scale) — if scale differs
3. **Affine 2D** (6 parameters) — if non-uniform scaling exists
4. **Projective** — if perspective effects are significant

### Rotation Decomposition

The final sitting-to-supine rotational relationship is NOT a single number from one algorithm. It decomposes as:

```
final_rotation = pentacam_to_elita_rotation + elita_cyclotorsion
```

Where:
- `pentacam_to_elita_rotation` = coordinate system transformation (unknown, to be estimated)
- `elita_cyclotorsion` = pre-dock→post-dock rotation (estimated by ELITA pipeline)

**This decomposition is critical for understanding what each component contributes.**

---

## 11. Cross-System Registration Contract

### Input

```python
CrossSystemRegistrationInput:
    pentacam: PentacamDetectionResult      # Pentacam sitting detection
    elita_supine: IrisDetectionResult       # ELITA supine detection
    elita_cyclotorsion: CorrespondenceResult # ELITA rotation estimate
    pentacam_coordinate_system: str         # "pentacam_pixel"
    elita_coordinate_system: str            # "elita_pixel"
```

### Output

```python
CrossSystemRegistrationResult:
    valid: bool
    failure: RegistrationFailureKind
    failure_reason: str

    # Transformation
    transformation_model: TransformationModel
    rotation_deg: float
    translation_x: float
    translation_y: float
    scale: float
    transform_matrix: Optional[np.ndarray]

    # Composition
    elita_cyclotorsion_deg: float
    final_sitting_to_supine_deg: Optional[float]

    # Quality
    n_correspondences: int
    n_inliers: int
    inlier_fraction: float
    residual_rms: float
    confidence: float
```

---

## 12. Failure Modes

| Failure Kind | Meaning | Behavior |
|-------------|---------|----------|
| `NO_PENTACAM` | No Pentacam detection available | Reject |
| `NO_ELITA` | No ELITA detection available | Reject |
| `INSUFFICIENT_PENTACAM_FEATURES` | Too few Pentacam features | Reject |
| `INSUFFICIENT_ELITA_FEATURES` | Too few ELITA features | Reject |
| `WEAK_CORRESPONDENCE` | Low correspondence quality | Reject |
| `AMBIGUOUS_CORRESPONDENCE` | Multiple competing hypotheses | Reject |
| `INCONSISTENT_TRANSFORMATION` | Transform estimates inconsistent | Reject |
| `EXCESSIVE_RESIDUAL` | High reprojection error | Reject |
| `COORDINATE_MISMATCH` | Incompatible coordinate systems | Reject |
| `IMAGE_QUALITY_FAILURE` | Poor image quality | Reject |
| `MISSING_METADATA` | Required metadata absent | Reject |

**Principle:** Honest rejection over confident incorrect registration.

---

## 13. Required Dataset

### Minimum Required

| Item | Format | Priority |
|------|--------|----------|
| Pentacam sitting images | Device export | REQUIRED |
| ELITA pre-dock sitting images | JPEG/PNG | REQUIRED |
| ELITA post-dock supine images | JPEG/PNG | REQUIRED |
| Eye pairing | JSON manifest | REQUIRED |
| Eye laterality | In manifest | REQUIRED |

### Preferred

| Item | Format | Priority |
|------|--------|----------|
| Reference rotation | Degrees | PREFERRED |
| Multiple patients | ≥3 | PREFERRED |
| Repeat captures | ≥2 per position | OPTIONAL |
| Device metadata | Acquisition info | OPTIONAL |

### Do NOT Require

- Patient-identifiable information
- Unanonymized data

---

## 14. Validation Methodology

### Dataset Split

| Split | Purpose | Size |
|-------|---------|------|
| Development | Algorithm tuning | Small |
| Validation | Threshold calibration | Medium |
| Holdout | Final evaluation | Remaining |

### Leakage Prevention

- No patient overlap between splits
- No eye overlap between splits
- No repeated capture leakage
- Final test set untouched during tuning

---

## 15. Metrics

### Registration Metrics

| Metric | Definition | Status |
|--------|-----------|--------|
| Rotation error | |θ̂ - θ_ref| | Requires reference |
| Translation error | ||t̂ - t_ref|| | Requires reference |
| Scale error | |ŝ - s_ref| | Requires reference |
| Reprojection residual | RMS pixel error | Computed |
| Inlier fraction | Inliers / total | Computed |
| Rejection rate | Fraction rejected | Computed |
| False-confidence rate | OK but wrong | Requires reference |
| Runtime | Processing time | Measured |

### Engineering Acceptance

| Criterion | Threshold |
|-----------|-----------|
| Deterministic | 100% |
| Bounded runtime | <500ms |
| Honest rejection | No false confidence |
| No catastrophic failure | Graceful degradation |

### Clinical Acceptance

**TBD / requires clinical-team definition.**

---

## 16. Implementation Boundary

| Item | Status |
|------|--------|
| ELITA pipeline | READY |
| Pentacam types | DESIGNED |
| Cross-system types | DESIGNED |
| Registration algorithm | NOT IMPLEMENTED (no data) |
| Real data validation | BLOCKED |
| Synthetic contract tests | IMPLEMENTED |

---

## 17. Current Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| No Pentacam images | Cannot implement detector | Obtain from clinical team |
| No Pentacam ↔ ELITA pairs | Cannot validate matching | Obtain paired data |
| Coordinate system unknown | Cannot design transform | Empirical measurement |
| ELITA pre/post-dock data unavailable | Cannot validate ELITA | Obtain from clinical team |

---

## 18. Recommended Next Phase

**BLOCKED on data acquisition.**

When real Pentacam + ELITA paired data becomes available:
1. Inspect Pentacam images
2. Determine coordinate system
3. Implement minimal Pentacam detector
4. Build cross-system correspondence
5. Estimate coordinate transformation
6. Validate with real paired data

---

## 19. Files Created

| File | Purpose |
|------|---------|
| `pupil_tracking/pentacam/cross_system.py` | CrossSystemRegistrationInput/Result types |
| `pupil_tracking/tests/test_pentacam_types.py` | 32 tests (22 Pentacam + 10 cross-system) |
| `IRIS_PHASE16_CROSS_SYSTEM_REGISTRATION_ARCHITECTURE.md` | This report |

**No production code modified.**

---

## 20. Production Safety

| Check | Status |
|-------|--------|
| UnifiedDetector unchanged | VERIFIED |
| Pupil detection unchanged | VERIFIED |
| Limbus detection unchanged | VERIFIED |
| Calibration unchanged | VERIFIED |
| GUI unchanged | VERIFIED |
| Production inference unchanged | VERIFIED |
| ELITA iris behavior unchanged | VERIFIED |
| No new dependency | VERIFIED |
| No clinical data committed | VERIFIED |
| Cross-system code isolated | VERIFIED |
| Not wired to production | VERIFIED |

---

## 21. Summary

| Item | Value |
|------|-------|
| Baseline | `0567a37` |
| ELITA contract | DESIGNED + IMPLEMENTED + TESTED |
| Pentacam contract | DESIGNED + SYNTHETIC TESTS ONLY |
| Cross-system contract | DESIGNED + SYNTHETIC TESTS ONLY |
| Real-data availability | BLOCKED — NOT AVAILABLE |
| Common representation | Geometry + feature hybrid |
| Coordinate findings | UNKNOWN (no data) |
| Transformation model | Rigid/Similarity/Affine 2D (progressive) |
| Tests | 135/135 PASS (103 iris + 32 Pentacam) |
| Production safety | PASS |
| Current blocker | Real Pentacam + ELITA paired data required |

**STOP.** Do NOT implement registration without data. Do NOT begin Phase XVII.

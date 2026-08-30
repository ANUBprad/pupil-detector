# IRIS Integration Architecture & Contract

> **Phase**: VII — Integration Architecture (planning only)
> **Date**: 2026-08-30
> **Repository**: https://github.com/ANUBprad/pupil-detector (branch `main`)
> **HEAD**: `3f882a5`
> **Status**: Architecture document. No code modified. No production changes.

---

## 1. Executive Summary

This document defines the architecture for integrating the existing iris
subsystem with the existing pupil/limbus detection pipeline. The goal is to
enable future pre-dock/post-dock cyclotorsion estimation without changing any
existing detection behavior.

**Key architectural decisions:**

- The iris subsystem consumes `EllipseParams` (pupil, limbus) already produced
  by the existing `UnifiedDetector`. No duplication of geometry.
- The iris subsystem returns `IrisDetectionResult` — a standalone type that
  does not modify `EyeDetectionResult`.
- Integration is **additive**: iris analysis appends to the result, never
  replaces or mutates existing fields.
- Iris processing is **optional and disabled by default**. When disabled, zero
  overhead is added to the existing pipeline.
- Failure isolation: an iris failure MUST NOT invalidate a valid
  `EyeDetectionResult`.

---

## 2. Clinical Motivation

The iris subsystem aims to estimate **cyclotorsion** — the angular rotation of
the eye when transitioning from upright (pre-dock) to supine (post-dock)
position. This information helps surgeons precisely correct astigmatism axis.

**Workflow:**
1. Acquire pre-dock image (upright, undocked)
2. Acquire post-dock image (supine, docked)
3. Detect iris features in both images
4. Match corresponding features
5. Estimate rotation angle
6. Report cyclotorsion to the surgeon

**Explicitly NOT in scope:**
- Astigmatism correction
- Treatment decisions
- Clinical claims
- Autonomous surgical guidance

---

## 3. Existing Detection Pipeline

### 3.1 Entry Point

**FACT**: The production detector is `UnifiedDetector` in
`pupil_tracking/core/detector.py`. It is called as:

```python
det = UnifiedDetector()
result = det.detect(image, frame_number=0, source='image.jpeg')
```

### 3.2 Output Type

**FACT**: The detector returns `EyeDetectionResult` (defined in
`pupil_tracking/utils/types.py:452`), which contains:

| Field | Type | Purpose |
|-------|------|---------|
| `pupil` | `PupilDetection` | Pupil ellipse, confidence, quality |
| `limbus` | `LimbusDetection` | Limbus ellipse, confidence, quality, WTW |
| `corneal_center` | `CornealCenterResult` | Specular reflection center |
| `calibration` | `CalibrationInfo` | px↔mm conversion |
| `metadata` | `FrameMetadata` | Timestamp, frame number, source, dimensions |
| `overall_quality` | `DetectionQuality` | SURGICAL/CLINICAL/RESEARCH/INSUFFICIENT |
| `overall_confidence` | `float` | Aggregate confidence [0,1] |
| `alerts` | `List[str]` | Warning messages |

### 3.3 Key Geometry Types

**FACT**: Both pupil and limbus are represented as `EllipseParams`
(`pupil_tracking/utils/types.py:170`):

```python
@dataclass
class EllipseParams:
    center_x: float
    center_y: float
    semi_major: float
    semi_minor: float
    angle_deg: float         # [0, 180), CCW from +x-axis
    # + uncertainty fields, fit quality, etc.
```

### 3.4 Convenience Properties

**FACT**: `EyeDetectionResult` provides:
- `has_pupil` → `bool` (pupil detected)
- `has_limbus` → `bool` (limbus detected)
- `has_both` → `bool` (both detected)

### 3.5 Quality Grades

**FACT**: `DetectionQuality` enum (`types.py:25`):
- SURGICAL (confidence ≥ 0.75)
- CLINICAL (≥ 0.55)
- RESEARCH (≥ 0.30)
- INSUFFICIENT (≥ 0.0)
- NO_DETECTION

---

## 4. Existing Iris Subsystem

### 4.1 Modules

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `iris/types.py` | 208 | Data contracts: IrisROI, IrisFeature, IrisFeatureSet, IrisDetectionResult | IMPLEMENTED |
| `iris/config.py` | 54 | IrisConfig: lattice, thresholds, insets | IMPLEMENTED |
| `iris/roi.py` | 179 | IrisROIExtractor: builds annular ROI from pupil/limbus | IMPLEMENTED |
| `iris/detect.py` | 184 | IrisFeatureDetector, detect_iris_features() | IMPLEMENTED |
| `iris/extraction.py` | 410 | IrisFeatureExtractor: lattice sampling, descriptors | IMPLEMENTED |
| `iris/normalization.py` | 137 | IrisNormalizer: (angle, radial_norm) ↔ pixel mapping | IMPLEMENTED |
| `iris/paired.py` | 199 | make_synthetic_pair(): synthetic A→B pairs | IMPLEMENTED |
| `iris/correspondence.py` | 994 | estimate_correspondence(), evaluate_pair() | IMPLEMENTED |

### 4.2 Single-Image Entry Point

**FACT**: The public entry point is `detect_iris_features()` in
`iris/detect.py:161`:

```python
def detect_iris_features(
    image: np.ndarray,           # BGR or grayscale
    pupil: Optional[EllipseParams],
    limbus: Optional[EllipseParams],
    *,
    config: Optional[IrisConfig] = None,
    reflection_remover: Optional[ReflectionRemover] = None,
    external_occlusion: Optional[np.ndarray] = None,
) -> IrisDetectionResult
```

### 4.3 Output Type

**FACT**: Returns `IrisDetectionResult` (`iris/types.py:191`):

```python
@dataclass
class IrisDetectionResult:
    valid: bool
    status: IrisStatus          # OK, NO_ROI, NO_FEATURES
    feature_set: IrisFeatureSet # ROI + features + stats
    mask_stats: Dict[str, float]
    processing_time_ms: float
```

### 4.4 Feature Set

**FACT**: `IrisFeatureSet` (`iris/types.py:164`):

```python
@dataclass
class IrisFeatureSet:
    roi: IrisROI
    features: List[IrisFeature]
    num_candidates: int
    num_accepted: int
    region_coverage: float
    usable_fraction: float
```

### 4.5 Correspondence Entry Point

**FACT**: The correspondence entry point is `estimate_correspondence()` in
`iris/correspondence.py:787`:

```python
def estimate_correspondence(
    image_a, image_b,
    feature_set_a, feature_set_b,
    baseline=MatchingBaseline.GEOMETRIC_DESCRIPTOR,
    rotation_method="consensus",
    config=None,
) -> CorrespondenceResult
```

---

## 5. Integration Boundary

### 5.1 The Clean Boundary

```
EXISTING PIPELINE                    IRIS SUBSYSTEM
─────────────────                    ──────────────
UnifiedDetector.detect()             detect_iris_features()
  → pupil: EllipseParams               → IrisROI (consumes pupil/limbus)
  → limbus: EllipseParams              → IrisFeatureSet (features, coords)
  → calibration: CalibrationInfo       → IrisDetectionResult
  → metadata: FrameMetadata
  → EyeDetectionResult                 estimate_correspondence()
                                         → CorrespondenceResult
```

### 5.2 Information Flow Direction

**PROPOSED**: The iris subsystem is a **downstream consumer** of existing
detection results. It reads geometry from `EyeDetectionResult` but never
writes to it.

```
Image → UnifiedDetector → EyeDetectionResult
                              ↓ (read only)
                         IrisFeatureDetector → IrisDetectionResult
                              ↓ (pair two results)
                         estimate_correspondence → CorrespondenceResult
```

---

## 6. Architecture

### 6.1 Conceptual Layers

```
Layer 0: EXISTING DETECTION
    UnifiedDetector
    PupilDetection, LimbusDetection
    CalibrationInfo, FrameMetadata
    EyeDetectionResult

Layer 1: IRIS FEATURE DETECTION (IMPLEMENTED)
    IrisFeatureDetector
    IrisROI, IrisFeature, IrisFeatureSet
    IrisDetectionResult

Layer 2: IRIS CORRESPONDENCE (IMPLEMENTED)
    estimate_correspondence()
    Correspondence, CorrespondenceResult

Layer 3: PRE/POST PAIRING (PROPOSED)
    AcquisitionStage enum
    IrisPairResult
    Pairing logic

Layer 4: CYCLOTORSION ESTIMATION (FUTURE)
    CyclotorsionResult
    Clinical integration
```

### 6.2 Dependency Rule

**PROPOSED**: Each layer depends only on layers below it. Layer 0 (existing
detection) has NO dependency on any iris layer. This ensures the existing
pipeline is never affected by iris code.

---

## 7. Data Flow

### 7.1 Single-Image Flow

```
ELITA image (BGR numpy array)
    │
    ▼
[EXISTING] UnifiedDetector.detect(image)
    │
    ├── PupilDetection (ellipse, confidence, quality)
    ├── LimbusDetection (ellipse, confidence, quality)
    ├── CalibrationInfo
    └── FrameMetadata
    │
    ▼
[EXISTING] EyeDetectionResult
    │
    ▼ (if iris enabled)
[IMPLEMENTED] detect_iris_features(image, pupil.ellipse, limbus.ellipse)
    │
    ├── IrisROI (from pupil/limbus geometry)
    ├── IrisFeatureSet (features with pixel + iris-relative coords)
    └── IrisDetectionResult
```

### 7.2 Pair Flow (Future)

```
Pre-dock image + Post-dock image
    │
    ▼
[EXISTING] Detect both → EyeDetectionResult_pre, EyeDetectionResult_post
    │
    ▼
[IMPLEMENTED] Detect iris in both → IrisDetectionResult_pre, IrisDetectionResult_post
    │
    ▼
[PROPOSED] Pair → IrisPairResult (stage metadata, eye identity)
    │
    ▼
[IMPLEMENTED] estimate_correspondence(feats_pre, feats_post)
    │
    └── CorrespondenceResult (rotation, scale, matches, quality)
```

### 7.3 Label Key

- **EXISTING**: Already implemented in the production pipeline
- **IMPLEMENTED**: Already implemented in the iris subsystem
- **PROPOSED**: Interface needed for integration (not yet implemented)
- **FUTURE**: Not implemented yet

---

## 8. Image-Level Contract

### 8.1 What the Iris Subsystem Consumes (EXISTING)

| Input | Source | Type | Required |
|-------|--------|------|----------|
| Image | Direct | `np.ndarray` (BGR or grayscale) | Yes |
| Pupil ellipse | `EyeDetectionResult.pupil.ellipse` | `EllipseParams` | Yes |
| Limbus ellipse | `EyeDetectionResult.limbus.ellipse` | `EllipseParams` | Yes |
| External occlusion mask | `SyntheticPair.occlusion_mask` | `np.ndarray` (bool) | No |

### 8.2 What the Iris Subsystem Produces (IMPLEMENTED)

```python
@dataclass
class IrisDetectionResult:
    """Single-image iris analysis result."""

    # Status
    valid: bool                          # True if features detected
    status: IrisStatus                   # OK, NO_ROI, NO_FEATURES

    # Feature data
    feature_set: IrisFeatureSet          # ROI + features + statistics

    # Diagnostics
    mask_stats: Dict[str, float]         # Usable fraction, occlusion stats
    processing_time_ms: float            # Wall-clock time
```

### 8.3 Feature Set Contents (IMPLEMENTED)

```python
@dataclass
class IrisFeatureSet:
    roi: IrisROI                         # Annular ROI geometry
    features: List[IrisFeature]          # Detected features
    num_candidates: int                  # Before quality filter
    num_accepted: int                    # After quality filter
    region_coverage: float               # Fraction of annulus covered
    usable_fraction: float               # Fraction not occluded/reflective
```

### 8.4 Feature Contents (IMPLEMENTED)

```python
@dataclass
class IrisFeature:
    id: int                              # Unique within the set

    # Image-pixel position
    x: float                             # Pixel x-coordinate
    y: float                             # Pixel y-coordinate

    # Iris-relative coordinates
    angle_deg: float                     # [0, 360), CCW from +x-axis
    radial_norm: float                   # (0, 1], pupil→limbus

    # Feature characteristics
    scale: float                         # Relative to iris outer radius
    orientation_deg: float               # Feature orientation
    feature_type: IrisFeatureType        # TEXTURE, CRYPT, FURROW, UNKNOWN

    # Quality
    response: float                      # Gradient energy response
    local_contrast: float                # Local intensity contrast
    visibility: float                    # [0, 1], occlusion fraction
    confidence: float                    # [0, 1], composite quality
    valid: bool

    # Descriptor
    descriptor: Optional[np.ndarray]     # 16-bin intensity histogram
```

### 8.5 Proposed: Extended Image-Level Result

**PROPOSED**: For integration, the image-level result should be wrapped with
acquisition metadata:

```python
@dataclass
class IrisAnalysisResult:
    """Extended result with acquisition context."""

    # Core result (IMPLEMENTED)
    detection: IrisDetectionResult

    # Acquisition context (PROPOSED)
    source_image_id: str                 # Unique image identifier
    acquisition_stage: AcquisitionStage  # PRE_DOCK, POST_DOCK, UNKNOWN
    eye_id: Optional[str]                # Eye identity (left/right)

    # Pipeline context (PROPOSED)
    detection_result: Optional[EyeDetectionResult]  # Parent detection
    image_width: int
    image_height: int
```

---

## 9. Pair-Level Contract

### 9.1 Proposed: Pair Result

**PROPOSED**: For pre/post-dock analysis:

```python
@dataclass
class IrisPairResult:
    """Result of comparing two iris feature sets."""

    # Source identity
    source_image_id: str                 # Image A identifier
    target_image_id: str                 # Image B identifier
    source_stage: AcquisitionStage       # PRE_DOCK
    target_stage: AcquisitionStage       # POST_DOCK
    eye_id: Optional[str]                # Same eye

    # Feature sets (IMPLEMENTED)
    feature_set_a: IrisFeatureSet
    feature_set_b: IrisFeatureSet

    # Correspondence (IMPLEMENTED)
    correspondence: CorrespondenceResult

    # Summary (PROPOSED)
    estimated_rotation_deg: float        # Cyclotorsion estimate
    estimated_scale: float               # Magnification change
    confidence: float                    # Overall pair confidence
    valid: bool                          # True if estimation succeeded
    failure_reason: str                  # Why it failed (if applicable)
```

### 9.2 CorrespondenceResult Contents (IMPLEMENTED)

```python
@dataclass
class CorrespondenceResult:
    valid: bool
    failure: FailureKind                 # OK, DEGENERATE, LOW_NCC, etc.
    failure_reason: str

    # Matching
    n_matches: int
    matched: List[Correspondence]        # Individual match pairs

    # Rotation
    estimated_rotation_deg: float
    coarse_rotation_deg: float
    rotation_estimates: Dict[str, float] # consensus, weighted_circular, ransac
    circular_std_deg: float
    consensus_fraction: float
    consensus_inlier_std_deg: float

    # Scale
    estimated_scale: float
    geometry_scale: float
    pupil_scale: float
    scale_valid: bool

    # Quality
    mean_ncc: float
    min_ncc: float
    ambiguity_ratio: float
    processing_time_ms: float
```

### 9.3 Individual Match (IMPLEMENTED)

```python
@dataclass
class Correspondence:
    index_a: int                         # Feature index in set A
    index_b: int                         # Feature index in set B
    angle_a: float                       # A-side angle
    angle_b: float                       # B-side angle
    radial_a: float                      # A-side radial norm
    radial_b: float                      # B-side radial norm
    confidence_a: float
    confidence_b: float
    weight_geometric: float
    weight_descriptor: float
    descriptor_distance: Optional[float]
    coarse_residual_deg: float
    refined_shift_deg: Optional[float]
    ncc: Optional[float]
    rotation_estimate_i: Optional[float]
```

---

## 10. Pre-Dock / Post-Dock Model

### 10.1 Acquisition Stage Enum

**PROPOSED**: Define acquisition stages for pairing:

```python
class AcquisitionStage(Enum):
    """When the image was acquired relative to surgical docking."""
    PRE_DOCK = "PRE_DOCK"      # Upright, undocked
    POST_DOCK = "POST_DOCK"    # Supine, docked
    UNKNOWN = "UNKNOWN"        # Stage not specified
```

### 10.2 Required Pairing Metadata

**PROPOSED**: To safely pair images, the following metadata is required:

| Metadata | Source | Required | Notes |
|----------|--------|----------|-------|
| Eye identity | `FrameMetadata.source` or manual | Yes | Must be same eye |
| Acquisition stage | `AcquisitionStage` | Yes | PRE_DOCK ↔ POST_DOCK |
| Image dimensions | `FrameMetadata` | Yes | Must match for correspondence |
| Pupil geometry | `EyeDetectionResult.pupil` | Yes | Required for iris ROI |
| Limbus geometry | `EyeDetectionResult.limbus` | Yes | Required for iris ROI |
| Timestamp | `FrameMetadata.timestamp` | No | Useful for ordering |

### 10.3 Pairing Rules

**PROPOSED**:
1. Images must be from the same eye (left/right)
2. One must be PRE_DOCK, the other POST_DOCK
3. Both must have valid pupil and limbus detection
4. Both must have valid iris feature sets (status=OK)
5. Image dimensions should match (or be rescalable)

### 10.4 What is NOT Required

**PROPOSED**: The following are NOT required for pairing:
- Exact timestamp matching
- Identical illumination
- Identical camera angle
- Identical pupil size (scale handles this)
- Ground truth rotation (that's what we're estimating)

---

## 11. Feature Representation

### 11.1 Coordinate Systems (IMPLEMENTED)

Each feature carries both coordinate systems:

**Image-pixel coordinates** (absolute):
- `x`, `y`: position in the source image pixel space

**Iris-relative coordinates** (scale-invariant):
- `angle_deg`: angular position [0, 360), CCW from +x-axis
- `radial_norm`: fractional position (0, 1] from pupil to limbus

### 11.2 Why Two Coordinate Systems (IMPLEMENTED)

**FACT** (from `normalization.py:1-12`):
> "The normalized radial coordinate is invariant to the absolute pixel size
> of the iris... two images of the same eye that differ in scale still place
> an anatomical point at the same (angle, radial) location"

- Iris-relative coordinates enable matching across scale changes
- Pixel coordinates enable visualization and pixel-space analysis
- Both are retained for flexibility

### 11.3 Feature Types (IMPLEMENTED)

| Type | Description | Detected By |
|------|-------------|-------------|
| TEXTURE | Generic local texture | Heuristic |
| CRYPT | Dark, contrasty pit | Anisotropy + darkness |
| FURROW | Elongated groove | Anisotropy + gradient |
| UNKNOWN | Unclassifiable | Fallback |

### 11.4 Descriptor (IMPLEMENTED)

**FACT**: The descriptor is a 16-bin normalized intensity histogram of the
local patch (5px radius). It is illumination-rough and scale-invariant by
design.

---

## 12. Correspondence Representation

### 12.1 Matching Baseline (IMPLEMENTED)

| Baseline | Weight Formula | Use Case |
|----------|---------------|----------|
| GEOMETRIC | `min(conf_a, conf_b)` | Pure geometry + confidence |
| GEOMETRIC_DESCRIPTOR | `min(conf_a, conf_b) * desc_sim` | Adds descriptor similarity |

**FACT**: Both baselines produce nearly identical results on surgical iris
images (Phase VI audit finding).

### 12.2 Rotation Estimation (IMPLEMENTED)

Three estimators are computed:
1. **Consensus**: Binned angular histogram, modal cluster mean (default)
2. **Weighted circular**: Full-set circular mean (sensitive to outliers)
3. **RANSAC**: Exhaustive two-point inlier consensus

### 12.3 Scale Estimation (IMPLEMENTED)

**FACT**: Scale is estimated as the median per-match pixel-radius ratio
(`correspondence.py:930-940`). Additionally, `geometry_scale` provides the
ROI-based scale from `limbus_radius_px` ratio.

---

## 13. Rotation Representation

### 13.1 Convention (IMPLEMENTED)

**FACT** (from `correspondence.py:26`):
> "a feature at iris angle phi in IMAGE A appears at iris angle phi - rot
> in IMAGE B when the applied OpenCV rotation is +rot (positive = clockwise
> on screen)"

- Positive rotation = clockwise on screen
- Estimated rotation in degrees [0, 360)
- Minimal circular difference used for error metric

### 13.2 Proposed: Cyclotorsion Result

**FUTURE**: The final clinical output:

```python
@dataclass
class CyclotorsionResult:
    """Estimated eye rotation between pre-dock and post-dock."""
    angle_deg: float                     # Rotation in degrees
    axis: str                            # "clockwise" or "counterclockwise"
    confidence: float                    # [0, 1]
    valid: bool                          # False if estimation failed
    failure_reason: str
```

---

## 14. Quality and Confidence

### 14.1 Feature-Level Confidence (IMPLEMENTED)

**FACT**: Feature confidence is computed as (`extraction.py:337-349`):
```python
confidence = 0.7 * resp + 0.3 * clr
```
where `resp` = response/(2*min_contrast), `clr` = boundary clearance.

### 14.2 Feature Set Quality Metrics (IMPLEMENTED)

| Metric | Range | Meaning |
|--------|-------|---------|
| `num_accepted` | 0–120 | More features = better correspondence |
| `region_coverage` | [0, 1] | Fraction of annulus covered |
| `usable_fraction` | [0, 1] | Fraction not occluded/reflective |

### 14.3 Correspondence Quality Metrics (IMPLEMENTED)

| Metric | Range | Meaning |
|--------|-------|---------|
| `mean_ncc` | [-1, 1] | Mean NCC of refined matches |
| `consensus_fraction` | [0, 1] | Fraction in consensus cluster |
| `consensus_inlier_std_deg` | [0, ∞) | Spread within consensus cluster |
| `circular_std_deg` | [0, ∞) | Global circular std of estimates |
| `ambiguity_ratio` | [0, 1] | Fraction of ambiguous matches |

### 14.4 Proposed: Pair-Level Confidence

**PROPOSED**: For integration, a composite pair confidence:

```python
pair_confidence = (
    0.4 * min(feature_conf_a, feature_conf_b) +
    0.3 * correspondence.consensus_fraction +
    0.2 * (1.0 - correspondence.circular_std_deg / 10.0) +
    0.1 * min(n_matches / 20.0, 1.0)
)
```

---

## 15. Failure Model

### 15.1 Iris Detection Failures (IMPLEMENTED)

| Status | Cause | Recovery |
|--------|-------|----------|
| `NO_ROI` | Missing/invalid pupil or limbus | Fix upstream detection |
| `NO_FEATURES` | ROI valid but no texture | Use different image |

### 15.2 Correspondence Failures (IMPLEMENTED)

| FailureKind | Cause | Precedence |
|-------------|-------|------------|
| `DEGENERATE` | < 4 matches | 1 (checked first) |
| `LOW_NCC` | > 50% refined NCC below 0.42 | 2 |
| `LOW_SIMILARITY` | > 50% descriptor similarity < 0.5 | 3 |
| `HIGH_RESIDUAL` | Consensus < 50% or std > 2.0° | 4 |
| `AMBIGUOUS` | > 50% ambiguous matches | 5 |
| `OK` | All gates passed | 6 (final) |

### 15.3 Proposed: Pairing Failures

| Failure | Cause | Recovery |
|---------|-------|----------|
| `DIFFERENT_EYE` | Images from different eyes | Re-pair correctly |
| `MISSING_STAGE` | Acquisition stage unknown | Manual annotation |
| `GEOMETRY_MISMATCH` | Pupil/limbus too different | Check image quality |

---

## 16. Failure Isolation

### 16.1 Principle

**The iris subsystem MUST fail independently without affecting the existing
pupil/limbus pipeline.**

### 16.2 Isolation Mechanism

**PROPOSED**: The iris subsystem is called AFTER the existing pipeline
completes. If iris analysis fails:

```python
# PROPOSED integration pattern
result = unified_detector.detect(image)  # Always completes
iris_result = None
if iris_enabled and result.has_both:
    try:
        iris_result = detect_iris_features(
            image, result.pupil.ellipse, result.limbus.ellipse
        )
    except Exception:
        iris_result = None  # Iris failure does not affect result
# result remains valid regardless of iris outcome
```

### 16.3 Failure Scenarios

| Scenario | Iris Effect | Existing Pipeline Effect |
|----------|------------|------------------------|
| Pupil detection fails | Iris: NO_ROI | None (pupil already failed) |
| Limbus detection fails | Iris: NO_ROI | None (limbus already failed) |
| Iris ROI invalid | Iris: NO_ROI | None |
| Too few features | Iris: NO_FEATURES | None |
| Correspondence fails | Result: failure != OK | None |
| Rotation ambiguous | Result: valid=False | None |
| Confidence too low | Result: valid=False | None |

### 16.4 Never Modify EyeDetectionResult

**CRITICAL**: The iris subsystem MUST NOT:
- Add fields to `EyeDetectionResult`
- Modify `pupil`, `limbus`, `calibration`, or `metadata`
- Change `overall_quality` or `overall_confidence`
- Alter `alerts`
- Affect `to_dict()` output

The iris result is a **separate** object returned alongside (not inside)
`EyeDetectionResult`.

---

## 17. Existing Pipeline Preservation

### 17.1 Safe Future Integration Points

| Integration Point | Location | Safety |
|-------------------|----------|--------|
| After `UnifiedDetector.detect()` returns | Caller code | SAFE — reads only |
| After `EyeDetectionResult` is constructed | Caller code | SAFE — does not modify |
| In a new wrapper function | New file | SAFE — additive |
| In the GUI as a separate panel | `gui_app.py` | SAFE — separate display |

### 17.2 DO NOT INTEGRATE HERE

| Location | Reason |
|----------|--------|
| Inside `UnifiedDetector.detect()` | Would change detection behavior |
| Inside `PupilDetection` or `LimbusDetection` | Would corrupt existing types |
| Inside `CalibrationInfo` | Would affect measurements |
| Inside `FrameMetadata` | Would alter frame processing |
| Inside `to_dict()` | Would change serialization |
| Inside `apply_smoothed_dict()` | Would affect Kalman smoothing |
| Inside video processing pipeline | Would slow down existing flow |
| Inside training pipeline | Would affect model training |

### 17.3 What Must NOT Change

- `UnifiedDetector.detect()` signature and behavior
- `EyeDetectionResult` fields and serialization
- `PupilDetection` and `LimbusDetection` types
- `CalibrationInfo` computation
- `FrameMetadata` structure
- Any existing test expectations
- Existing GUI behavior
- Video processing pipeline
- ONNX model loading/inference

---

## 18. Future Integration Points

### 18.1 Phase VIII: Correspondence Improvements

**PROPOSED**: Improve the correspondence layer based on Phase VI audit:
- Fix translation FALSE-OK
- Tighten consensus_inlier_std_max_deg
- Add false-OK metric to acceptance criteria
- Extend rotation search window

### 18.2 Phase IX: Real ELITA Validation

**PROPOSED**: Run the identical harness on first real ELITA paired data:
- Validate synthetic benchmark conclusions
- Measure real-world performance
- Identify clinical-grade requirements

### 18.3 Phase X: Cyclotorsion Estimation

**FUTURE**: Implement the clinical output:
- Pair pre-dock and post-dock iris results
- Run correspondence
- Report rotation angle with confidence

### 18.4 Phase XI: Doctor-Facing Integration

**FUTURE**: Surface cyclotorsion to the surgeon:
- Add to GUI as a separate panel
- Display rotation angle and confidence
- Provide clinical interpretation (NOT treatment decisions)

### 18.5 Phase XII: End-to-End Validation

**FUTURE**: Complete system validation:
- Full pipeline from image acquisition to cyclotorsion report
- Clinical workflow integration
- Performance benchmarks

---

## 19. Prohibited Integration Points

### 19.1 Never Modify Existing Types

```python
# WRONG: Adding iris to EyeDetectionResult
@dataclass
class EyeDetectionResult:
    pupil: PupilDetection
    limbus: LimbusDetection
    iris: IrisDetectionResult  # ← PROHIBITED

# RIGHT: Separate result
result = unified_detector.detect(image)
iris_result = detect_iris_features(image, ...)
```

### 19.2 Never Mutate Existing Results

```python
# WRONG: Modifying the existing result
result = unified_detector.detect(image)
result.iris = iris_result  # ← PROHIBITED (field doesn't exist)

# RIGHT: Return separate objects
result = unified_detector.detect(image)
iris_result = detect_iris_features(image, ...)
# Both are independent; caller manages both
```

### 19.3 Never Add Dependencies to Existing Code

```python
# WRONG: Importing iris in detector.py
from pupil_tracking.iris.detect import detect_iris_features  # ← PROHIBITED

# RIGHT: Caller imports both independently
from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.detect import detect_iris_features
```

---

## 20. Real ELITA Data Requirements

### 20.1 Required Data

| Item | Required | Purpose |
|------|----------|---------|
| Pre-dock image | Yes | Source for feature extraction |
| Post-dock image | Yes | Target for feature extraction |
| Same-eye pairing | Yes | Must be same eye (left/right) |
| Sufficient iris visibility | Yes | ROI must be valid in both |
| Appropriate acquisition | Yes | Surgical microscope or slit lamp |

### 20.2 Optional Data

| Item | Required | Purpose |
|------|----------|---------|
| Acquisition metadata | No | Helps pairing but not required |
| Ground truth rotation | No | For validation only |
| Pupil/limbus annotations | No | Detector provides these |
| Timestamp | No | Useful for ordering |

### 20.3 What Proxy Data Cannot Validate

**FACT**: The synthetic benchmark uses clinical proxy images (surgical eye
images, not ELITA pre/post-dock captures). Proxy data validates:
- Algorithm correctness
- Rotation recovery accuracy
- Scale estimation
- Perturbation robustness

Proxy data CANNOT validate:
- Clinical performance
- Real-world ELITA image quality
- Actual cyclotorsion range
- Surgical workflow integration
- Doctor-facing usability

### 20.4 Data Acquisition Protocol (PROPOSED)

1. Capture pre-dock image (upright, undocked eye)
2. Capture post-dock image (supine, docked eye)
3. Record eye identity (left/right)
4. Record acquisition stage
5. Run detection pipeline on both
6. Run iris analysis on both
7. Pair and estimate correspondence
8. Report cyclotorsion

---

## 21. Phase Boundaries

### 21.1 Completed Phases

| Phase | Scope | Status |
|-------|-------|--------|
| I | Iris feature detection baseline | COMPLETE |
| II | Robustness evaluation | COMPLETE |
| III | Smoothed-Sobel hardening | COMPLETE |
| IV | Synthetic correspondence + rotation recovery | COMPLETE |
| V | Real ELITA validation plan | COMPLETE |
| V-A | Phase IV synthetic benchmark execution | COMPLETE |
| VI | Benchmark failure audit | COMPLETE |
| VII | Integration architecture & contracts | THIS DOCUMENT |

### 21.2 Future Phases

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| VIII | Targeted correspondence improvements | Phase VI audit findings |
| IX | Real ELITA pre/post paired-data validation | Real ELITA data |
| X | Validated cyclotorsion estimation | Phase IX |
| XI | Doctor-facing integration | Phase X |
| XII | End-to-end validation | Phase XI |

### 21.3 Phase Rules

1. Each phase is one discrete, well-defined unit
2. No phase modifies existing detection behavior
3. Each phase is committed and pushed separately
4. Each phase runs the full test suite
5. Each phase produces a report document

---

## 22. Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Real ELITA images too different from proxy | HIGH | Medium | Phase IX validates on real data |
| Correspondence fails on real iris texture | HIGH | Medium | Phase VIII improvements |
| Insufficient feature count on real images | MEDIUM | Medium | Alternative ROI sources (ONNX mask) |
| Integration breaks existing pipeline | HIGH | Low | Strict isolation rules (§16–17) |
| Clinical workflow integration困难 | MEDIUM | High | Phase XI focuses on usability |
| False-OK on clinical data | MEDIUM | Medium | Phase VIII tightening |

---

## 23. Open Questions

1. **What is the minimum feature count for reliable correspondence on real ELITA images?** The proxy benchmark shows 20–26 features are marginal for ±5–6 deg rotations. Real ELITA images may have different texture characteristics.

2. **Should the integration be called from `UnifiedDetector` or from the caller?** Architecture says caller (§16.2), but this means every caller must import iris code separately. Consider a thin wrapper that optionally includes iris.

3. **How should PRE_DOCK/POST_DOCK be annotated?** Manual? Automatic from image metadata? This affects the pairing contract.

4. **What if only one image has valid iris features?** The system must handle asymmetric availability gracefully.

5. **Should the iris result be serialized in `to_dict()`?** Architecture says no (§17.2), but the GUI may need it. Consider a separate serialization path.

6. **What is the acceptable rotation accuracy for clinical use?** The proxy benchmark shows ≤0.25 deg on well-featured images, ≤2.3 deg on sparse images. Clinical requirement TBD.

---

## 24. Recommended Next Phase

### Phase VIII: Targeted Correspondence Improvements

**Objective**: Fix the specific issues identified in the Phase VI audit:
1. Fix translation FALSE-OK (2/10 cases)
2. Add false-OK metric to acceptance criteria
3. Investigate sub-lattice NCC bias (root cause of all FALSE-OK)

**Scope**:
- `pupil_tracking/iris/correspondence.py` — acceptance logic
- `scripts/iris_phase4_correspondence_eval.py` — harness additions
- `pupil_tracking/tests/test_iris_correspondence.py` — new tests

**Non-goals**:
- Real ELITA validation (Phase IX)
- Cyclotorsion estimation (Phase X)
- GUI integration (Phase XI)
- Any modification to existing detection pipeline

**Verification**:
- 59/59 iris tests pass
- FALSE-OK count reduced from 10 to ≤2
- Translation FALSE-OK eliminated
- No regression on correct estimates

---

*This is an architecture document. No code was modified. No clinical claims
are made. The existing detection pipeline is fully preserved.*

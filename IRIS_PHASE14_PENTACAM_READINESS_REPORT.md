# IRIS PHASE XIV — PENTACAM DETECTION & CROSS-SYSTEM READINESS

**Date:** 2026-09-01
**HEAD:** `f7f3e13`
**Scope:** Pentacam readiness analysis + cross-system architecture design. No algorithm implementation.

---

## 1. Objective

Determine what information can reliably be extracted from a Pentacam sitting image that can eventually be matched with the ELITA supine representation. Establish the Pentacam-side representation before attempting cross-system registration.

---

## 2. Overall Project Sequence

```
ELITA pre-dock (sitting)
      +
ELITA post-dock (supine)
      ↓
Iris feature detection
      ↓
Iris feature correspondence
      ↓
Cyclotorsion / rotation angle
      ↓
Confidence / evidence validation
      ↓
Pentacam sitting image              ← WE ARE HERE
      +
ELITA supine information
      ↓
Future cross-system matching
      ↓
Sitting-to-supine rotational relationship
```

---

## 3. Current ELITA Capability

**FACT:** The ELITA iris/cyclotorsion pipeline is implemented:

| Component | Status |
|-----------|--------|
| Iris ROI extraction | READY |
| Feature extraction (72-angle × 8-radius) | READY |
| Correspondence (coarse + NCC) | READY |
| Rotation estimation (global_hybrid) | READY |
| Evidence gate | READY (disabled by default) |
| Validation harness | READY |
| Real ELITA paired data | **BLOCKED** |

---

## 4. Pentacam Data Availability

**BLOCKED — REAL PENTACAM DATA UNAVAILABLE.**

Search results:
- No files matching `*pentacam*` or `*scheimpflug*`
- No mentions of Pentacam in code (only in documentation/reports)
- `clinical_data/` contains surgical eye images, not Pentacam data
- All 17 grep hits for "Pentacam" are in documentation files

**Classification:**
- VERIFIED PENTACAM: 0
- POSSIBLE PENTACAM: 0
- NOT PENTACAM: 12 images (surgical eye photographs)
- UNKNOWN: 0

---

## 5. Pentacam Input Requirements

### What a Pentacam Device Produces

A Pentacam device captures Scheimpflug images of the anterior segment. It typically provides:

| Output | Description | Relevance |
|--------|-------------|-----------|
| Scheimpflug cross-section images | Narrow slit images of anterior segment | Direct image data |
| Corneal elevation maps | Anterior/posterior surface height | Topographic data |
| Pachymetry maps | Corneal thickness | Thickness data |
| Keratometry | Corneal curvature (K1, K2) | Curvature data |
| Anterior chamber depth | AC depth | Structural data |
| Axial/tangential curvature maps | Curvature maps | Topographic data |

### What We Likely Need for Cross-System Matching

For matching Pentacam sitting with ELITA supine, the most relevant inputs are:

| Input | Priority | Rationale |
|-------|----------|-----------|
| Scheimpflug cross-section image | HIGH | Contains visible iris/pupil/limbus geometry |
| Corneal elevation map | MEDIUM | May provide stable landmarks |
| Raw intensity image | HIGH | For feature extraction |

### Unknown Requirements

- What image format does the Pentacam export?
- What coordinate system does the Pentacam image use?
- Is the Pentacam image a full-face image or a narrow slit?
- What resolution is typical?
- Is the pupil/limbus visible in Pentacam images?

---

## 6. Pentacam Anatomical Candidates

Structures that could potentially connect Pentacam sitting with ELITA supine:

| Structure | Visibility in Pentacam | Stability | Geometric Usefulness | Correspondence Value | Limitations |
|-----------|----------------------|-----------|---------------------|---------------------|-------------|
| **Pupil boundary** | LIKELY | HIGH | Center + radius | HIGH | May be partially visible in slit images |
| **Limbus boundary** | LIKELY | HIGH | Center + radius + shape | HIGH | May be partial arc in slit images |
| **Iris texture** | UNCERTAIN | HIGH | Feature matching | HIGH | Pentacam illumination differs from ELITA |
| **Corneal apex** | YES | HIGH | Reference point | MEDIUM | Different coordinate system |
| **Anterior chamber** | YES | HIGH | Structural reference | LOW | Not directly visible in ELITA |
| **Scleral boundary** | UNCERTAIN | MEDIUM | Outer reference | LOW | May not be visible |

### Most Promising Candidates

1. **Pupil boundary** — Likely visible, stable, geometrically simple
2. **Limbus boundary** — Likely visible, stable, defines iris extent
3. **Iris texture/features** — Potentially powerful if illumination is compatible

---

## 7. Feature Representation

### Current ELITA Iris Representation

**FACT:** The ELITA iris representation uses:
- Annular ROI between pupil and limbus ellipses
- 72-angle × 8-radius lattice (5° angular step)
- 16-bin intensity histogram descriptors
- Iris-relative coordinates (angle_deg, radial_norm)
- Global coordinate system: image pixel space

### Proposed Pentacam Feature Representation

The Pentacam representation should be:
- **Isolated** — not dependent on ELITA representation
- **Deterministic** — same input produces same output
- **Structured** — carries geometry + descriptors + confidence

**Recommended approach (pending real data):**

1. **Detect pupil boundary** — ellipse fitting (reuse existing SmartContourFitter concepts)
2. **Detect limbus boundary** — ellipse fitting
3. **Extract iris features** — adapted lattice within detected annulus
4. **Compute descriptors** — local texture patches

### Key Differences from ELITA

| Aspect | ELITA | Pentacam (Expected) |
|--------|-------|-------------------|
| Image type | Surgical microscope | Scheimpflug slit |
| Illumination | Surgical light | Pentacam LED |
| Viewing angle | En-face | Cross-sectional |
| Coordinate system | Image pixel space | Device-specific |
| Iris visibility | Full annulus | Potentially partial arc |
| Reflection pattern | Surgical reflections | Pentacam-specific artifacts |

---

## 8. Coordinate-System Analysis

### ELITA Coordinate System

**FACT:** ELITA uses:
- Image pixel coordinates (x, y from top-left)
- Ellipse angle convention: counter-clockwise from positive x-axis, [0, 180)
- Iris-relative: angle_deg (CCW from x-axis), radial_norm (0=pupil, 1=limbus)
- Scale: pixels (dependent on image resolution and zoom)

### Pentacam Coordinate System (Expected)

**UNKNOWN:** Pentacam coordinate system depends on:
- Device model and software version
- Export format (image vs. data)
- Whether raw or processed images are used

**Assumptions that must be verified:**
- Pentacam images are likely in device-specific pixel coordinates
- Scheimpflug images have a different geometric projection than en-face images
- The optical distortion model differs from surgical microscope images

### Potential Transformations Between Systems

| Transformation | Likelihood | Notes |
|---------------|------------|-------|
| Translation | CERTAIN | Different imaging devices |
| Rotation | LIKELY | Different mounting/positioning |
| Scale | LIKELY | Different magnification |
| Perspective | POSSIBLE | Different viewing angle |
| Optical distortion | CERTAIN | Different lens systems |
| Non-rigid deformation | POSSIBLE | Different pressure/position |
| Reflection/occlusion | CERTAIN | Different illumination |

### Critical Insight

**The Pentacam ↔ ELITA matching problem is NOT a simple 2D rotation.** It involves:
- Different imaging geometries (en-face vs. cross-sectional)
- Different coordinate systems
- Different illumination patterns
- Potentially different visible structures

This phase identifies the problem complexity before implementing the registration algorithm.

---

## 9. Cross-System Matching Requirements

### Information Needed by Future Matcher

| Information | Status | Source |
|-------------|--------|--------|
| Pentacam feature coordinates | REQUIRES PENTACAM DATA | Future detection |
| Pentacam feature descriptors | REQUIRES PENTACAM DATA | Future detection |
| Pentacam anatomical geometry | REQUIRES PENTACAM DATA | Future detection |
| ELITA feature coordinates | AVAILABLE NOW | IrisFeatureSet |
| ELITA feature descriptors | AVAILABLE NOW | IrisFeatureSet |
| ELITA rotation estimate | AVAILABLE NOW | CorrespondenceResult |
| ELITA confidence/evidence | AVAILABLE NOW | CorrespondenceResult |
| Anatomical reference geometry | REQUIRES BOTH | Future analysis |

### Availability Classification

| Item | ELITA | Pentacam | Cross-System |
|------|-------|----------|-------------|
| Pupil geometry | AVAILABLE | REQUIRES DATA | REQUIRES BOTH |
| Limbus geometry | AVAILABLE | REQUIRES DATA | REQUIRES BOTH |
| Iris features | AVAILABLE | REQUIRES DATA | REQUIRES BOTH |
| Feature descriptors | AVAILABLE | REQUIRES DATA | REQUIRES BOTH |
| Rotation estimate | AVAILABLE | N/A | AVAILABLE |
| Confidence | AVAILABLE | N/A | AVAILABLE |
| Coordinate transform | N/A | N/A | UNKNOWN |

---

## 10. Dataset Requirements

### Minimum Required for Development

| Item | Format | Priority |
|------|--------|----------|
| Pentacam sitting images | Device export format | REQUIRED |
| Eye laterality | Metadata | REQUIRED |
| Corresponding ELITA case | Pairing information | REQUIRED |

### For Validation

| Item | Format | Priority |
|------|--------|----------|
| Pentacam sitting | Image | REQUIRED |
| ELITA pre-dock sitting | Image | REQUIRED |
| ELITA post-dock supine | Image | REQUIRED |
| Reference rotation | Degrees | PREFERRED (may not exist) |
| Repeat captures | Multiple per position | OPTIONAL |

---

## 11. Detection Implementation

**NOT IMPLEMENTED.** No Pentacam data is available to implement against.

If Pentacam data becomes available, the recommended minimal detector would:
1. Load Pentacam image
2. Detect pupil boundary (ellipse fitting)
3. Detect limbus boundary (ellipse fitting)
4. Extract iris features within annulus
5. Return structured PentacamDetectionResult

The detector should:
- Be deterministic
- Expose confidence/quality
- Return structured results
- Remain isolated from UnifiedDetector
- Have unit tests

---

## 12. Detection Results

**NOT APPLICABLE.** No Pentacam data was available to test against.

---

## 13. Validation Methodology

For future Pentacam validation:

1. **Determinism** — Same image produces identical results
2. **Geometry plausibility** — Pupil/limbus sizes within physiological range
3. **Feature quality** — Sufficient angular coverage for matching
4. **Runtime** — Within processing budget
5. **Cross-system matching** — Requires paired ELITA+Pentacam data

---

## 14. Runtime

**NOT MEASURED.** No Pentacam detector implemented.

Budget recommendation: <200ms for detection + feature extraction (similar to ELITA iris pipeline).

---

## 15. Testing

**STATUS:** 103/103 iris tests pass (unchanged).

If Pentacam code is added:
- Add deterministic synthetic fixtures for software contract testing
- Label all synthetic results as SYNTHETIC
- Never present synthetic results as Pentacam performance

---

## 16. ELITA Handoff Contract

### What Phase XIII Provides to Future Cross-System Stage

```python
# From ELITA pipeline
{
    "supine_image": np.ndarray,          # ELITA supine image
    "supine_feature_set": IrisFeatureSet, # features from supine image
    "cyclotorsion_estimate_deg": float,   # rotation from pre→post
    "rotation_confidence": str,           # "OK" / "LOW_EVIDENCE" / ...
    "evidence": {
        "feature_count": int,
        "angular_coverage": float,
        "global_inlier_fraction": float,
    }
}
```

### What Pentacam Stage Must Provide

```python
# From Pentacam detection (future)
{
    "sitting_image": np.ndarray,          # Pentacam sitting image
    "pupil_ellipse": EllipseParams,       # detected pupil geometry
    "limbus_ellipse": EllipseParams,      # detected limbus geometry
    "feature_set": IrisFeatureSet,        # extracted iris features
    "detection_confidence": str,          # quality assessment
    "coordinate_system": str,             # coordinate system identifier
}
```

---

## 17. Future Pentacam ↔ ELITA Matching

### Matching Problem Definition

Given:
- Pentacam sitting features (in Pentacam coordinate system)
- ELITA supine features (in ELITA coordinate system)
- ELITA cyclotorsion estimate (pre→post rotation)

Estimate:
- Coordinate transformation between Pentacam and ELITA systems
- Combined sitting-to-supine rotational relationship

### Matching Approach (Future)

1. **Feature correspondence** — Match Pentacam features to ELITA features
2. **Geometric transformation estimation** — Estimate 2D (or 2D+scale) transform
3. **Rotation composition** — Combine coordinate transform with ELITA cyclotorsion
4. **Confidence assessment** — Evaluate match quality

### Critical Unknown

**The relationship between Pentacam and ELITA coordinate systems is UNKNOWN.** This must be determined from real paired data before a matching algorithm can be implemented.

---

## 18. Current Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| No Pentacam images | CANNOT implement detector | Obtain from clinical team |
| No Pentacam ↔ ELITA pairs | CANNOT validate matching | Obtain paired data |
| Coordinate system relationship unknown | CANNOT design transform | Empirical measurement needed |
| Pentacam image format unknown | CANNOT design loader | Obtain sample images |

---

## 19. Production Safety

| Check | Status |
|-------|--------|
| UnifiedDetector unchanged | VERIFIED |
| Pupil detection unchanged | VERIFIED |
| Limbus detection unchanged | VERIFIED |
| Calibration unchanged | VERIFIED |
| GUI unchanged | VERIFIED |
| Production inference unchanged | VERIFIED |
| No new dependency | VERIFIED |
| No clinical data committed | VERIFIED |
| Iris remains additive | VERIFIED |
| Pentacam code not wired to production | VERIFIED |

---

## 20. Next Recommended Phase

**BLOCKED on Pentacam data.**

When Pentacam data becomes available:
1. Build minimal Pentacam image loader
2. Implement pupil/limbus detection (adapt existing concepts)
3. Extract iris features
4. Measure detection quality and runtime
5. Begin cross-system matching investigation

---

## 21. Final Verdict

**STATUS: READY FOR PENTACAM DATA ACQUISITION.**

The project has:
- Complete ELITA iris/cyclotorsion pipeline (103/103 tests)
- Pentacam representation designed (pending real data validation)
- Cross-system matching problem defined
- Coordinate-system complexity identified
- Production safety maintained

The only blocker is **real Pentacam data**. Once available, the minimal detector can be built and the cross-system matching investigation can begin.

**DO NOT:**
- Implement Pentacam detector without real data
- Claim cross-system registration works
- Modify production detector
- Begin Phase XV

---

## 22. Files Changed

**NONE.** This phase is analysis and documentation only. No code was modified.

---

## 23. Summary

| Item | Value |
|------|-------|
| Baseline | `f7f3e13` |
| Pentacam data | BLOCKED — NOT AVAILABLE |
| Algorithm changes | NONE |
| Production safety | PASS |
| Tests | 103/103 PASS (unchanged) |
| Current blocker | Real Pentacam images required |
| Future scope | Pentacam detection + cross-system matching |

**STOP.** Do NOT implement Pentacam detector. Do NOT begin Phase XV.

# Iris Feature Detection — Architecture & Feasibility Study

> **Phase**: Planning / architecture only — **no implementation**.
> **Repository**: https://github.com/ANUBprad/pupil-detector
> **Document date**: 2026-08-28
> **Status**: Proposed architecture; NOT yet implemented.

---

## 1. Executive Summary

This document is a complete architecture and feasibility study for the **iris-feature-detection**
phase of the Pupil-Limbus detector. It is planning-only: **no production code, model, or
dependency is changed or created** in this phase.

The immediate goal is **iris feature detection from ELITA pre-dock and post-dock images**.
Feature matching, registration and cyclotorsion estimation are **subsequent phases** that are
designed for (their interfaces are specified here) but are **not** implemented now.

### Critical data findings (read before anything else)

Two planning preconditions referenced in the task setup could not be satisfied from the
repository, and this report is intentionally honest about that:

1. **No `files/` requirements directory exists** in the repository. No per-phase requirement
   documents were available; this report uses the detailed task specification, `CLAUDE.md`,
   `opencode.md`, `gemini.md`, `README.md`, and the inspected code as its constraints.

2. **No ELITA pre-dock / post-dock paired images exist in the repository.** The only ophthalmic
   image data present is the existing **surgical pupil/limbus detection dataset** (suction-ring
   "docked"/"pre-docked" eye images) from the Cornea-Pupil-Limbus detection pipeline. Therefore
   **PHASE B (ELITA data findings) cannot report on real ELITA images**; instead this document
   reports quantitative evidence about the only available eye imagery and marks every
   ELITA-specific claim as **not yet established** (see §5 and §21).

The recommended path is therefore a **two-stage validation approach**: (a) validate the
iris-feature pipeline on the existing surgical imagery as a **proxy** to de-risk the architecture,
and (b) obtain and re-validate on **real paired ELITA pre/post-dock imagery** before the feature
matching / registration phases. Numerical acceptance thresholds that cannot be justified from
available data are explicitly marked **"to be established experimentally"**.

---

## 2. Clinical Motivation

The clinical objective is to support **precise astigmatism treatment-axis alignment** during
refractive surgery by compensating for **ocular cyclotorsion** (torsional/cyclorotational eye
movement).

When the patient changes from a **sitting (pre-dock)** position to a **supine (post-dock)**
position, the eye can rotate torsionally. If **stable anatomical iris features** are identified
before and after docking, they act as natural reference landmarks. Matching corresponding iris
features between the two positions allows the system to estimate the angle (θ) through which the
eye rotated, which in turn supports compensation of the astigmatism treatment axis.

The intended end-to-end pipeline is:

```
ELITA PRE-DOCK IMAGE ---> Iris region ---> Iris feature detection ---> Stable features ---+
                                                                                          |
ELITA POST-DOCK IMAGE --> Iris region ---> Iris feature detection ---> Stable features ---+---> Feature correspondence
                                                                                                  |
                                                                                                  v
                                                                                          Iris registration
                                                                                                  |
                                                                                                  v
                                                                                          Ocular rotation angle θ
                                                                                                  |
                                                                                                  v
                                                                                        Cyclotorsion compensation
                                                                                                  |
                                                                                                  v
                                                                                Astigmatism treatment-axis alignment
```

**Important framing:** iris feature detection is the **foundation** for the registration /
cyclotorsion stage. It is **not** the final clinical objective in itself. The detector must
therefore be designed so that a later registration stage can consume its output (§14–§16).

---

## 3. Current Project State

- **Branch / remote state**: local `main` at `b2f7cc4`, which matches `target`(ANUBprad/pupil-detector) `main`.
  Remotes: `origin` → `Shashwat-911/Cornea-Pupil-Limbus-detection-.git` (upstream, other account);
  `fork` → `ANUBprad/Cornea-Pupil-Limbus-detection-.git`; `target` → `ANUBprad/pupil-detector.git`
  (the authoritative project repository for this work).
- Working tree is **dirty** (many unrelated uncommitted modifications). This planning phase does
  **not** touch those files.
- **Production detection pipeline** (verified by code inspection):
  1. Image load (OpenCV)
  2. Grayscale handling (auto/force/off, CLAHE)
  3. Ring detection → docked / pre-docked classification
  4. Ring-aware adaptive preprocessing
  5. ML segmentation (ONNX Runtime production / PyTorch dev / FastInference video)
  6. Contour fitting (SmartContourFitter, classical CV fallback)
  7. Cross-validation and rejection
  8. Auto-calibration (limbus-anchored px→mm)
  9. Corneal-centre computation
  10. Quality grading
- **Successful / known state**: `UnifiedDetector.detect()` is the main orchestrator. Test suite
  baseline is 242/243 passing with **one documented pre-existing failure**
  (`test_eye_01_unchanged_after_ring_constraint`, containing hardcoded old-model expectations).
- **Gap for iris work**: there is currently **no** iris region extraction, iris feature
  detector, feature representation, matching, registration, or cyclotorsion code anywhere in the
  repository (verified by search — no "iris feature", "cyclotorsion", "registration", or "ELITA"
  matches outside the surgical docked/pre-docked suction-ring concept).

---

## 4. Existing Pipeline Reuse

The existing pipeline is rich and directly reusable for iris work. Evidence from code inspection:

| Concern | Existing asset | Location | Reusable for iris phase |
|---|---|---|---|
| Image load | `cv2.imread` | GUI / detector | Yes (same input) |
| Grayscale + CLAHE | `GrayscaleHandler` | `preprocessing/grayscale_handler.py` | Yes, for normalization |
| Reflection handling | `ReflectionRemover`, `reflection_removal.py` | `preprocessing/` | Yes — reuse to mask/remove specular reflections |
| Ring-aware adaptive preprocess | `RingAwarePreprocessor` | `preprocessing/ring_aware.py` | Possibly (docked vs pre-docked) |
| Pupil/limbus geometry | `EllipseParams` (center, semi-axes, angle, confidence) | `utils/types.py` | **Yes** — defines the iris annulus ROI |
| Segmentation masks | ONNX `masks={'pupil','iris'}`, `_clean_mask` | `ml/onnx_inference.py` | **Yes** — alternative iris-region source |
| Ellipse fitting | `SmartContourFitter`, `ellipse_fitter.py` | `core/` | Yes for ROI geometry |
| Confidence / quality | `confidence.py`, `DetectionQuality`, `QualityLevel` | `core/`, `utils/types.py` | **Yes** — pattern to mirror for feature confidence |
| Coordinate system | Image-pixel center coords; ellipse semi-axes in px | throughout | **Yes** — keep pixel space for features |
| Masking utilities | mask generation/cleaning (`ml/postprocess.py`) | `ml/` | Yes for occlusion/reflection masks |
| Overlay / visualization | `video_overlay.py`, GUI `drawing_mixin.py` | `video/`, `interface/` | **Yes** — extend for feature visualization |
| Video / camera path | `optimized_processor.py`, `video_processor.py` | `video/` | Future / optional |
| Testing structure | `pupil_tracking/tests/` (pytest, clinical/audit suites) | `tests/` | Yes |

### Execution path to trace for integration

```
ELITA image
  → cv2.imread
  → grayscale handling (GrayscaleHandler)
  → ring detection → docked/pre-docked
  → ring-aware preprocessing
  → ML segmentation (ONNX) → pupil/iris masks
  → contour fitting (SmartContourFitter) → pupil/limbus EllipseParams
  → cross-validation
  → calibration (limbus-anchored px→mm)
  → corneal centre
  → quality grade
  → EyeDetectionResult
  → GUI visualization / export
```

**Where iris features should integrate (future, not now):** after pupil/limbus geometry and the
iris mask are available — i.e. conceptually between the "segmentation + fitting" step and the
"final result" step — the iris-feature module would consume the **limbus/pupil ellipse + iris
mask** and produce an **`IrisFeatureSet`** (an attachment to the result). Because the existing
result is a dataclass, adding an optional `iris`/`features` field is a **future, non-breaking
extension** requiring no change to existing consumers.

---

## 5. ELITA Data Findings

### 5.1 Status of ELITA data — NOT AVAILABLE

- **No ELITA images, metadata, annotations, or paired pre/post-dock sets exist in the
  repository** (confirmed by full recursive listing and content search).
- The available ophthalmic data is the existing **surgical pupil/limbus detection dataset**:
  - `clinical_data/clean/` — 12 curated surgical eye images (eye_01–eye_14, no eye_05) with
    suction rings.
  - `clinical_data/training_data/images/` — 145 `frame_*.jpg` + a few `eye_XX.jpg` surgical
    frames; `masks/` — 253 mask images.
  - `clinical_data/annotations/` — metadata + production annotations JSON.
- These are **not** labeled pre-dock/post-dock ELITA captures and are not necessarily the same
  imaging modality.

### 5.2 Quantitative evidence from the available images (computed, not inferred)

The following statistics were measured directly on the available images (read-only analysis) and
are reported as **evidence about the only available imagery, not about ELITA**:

**Whole-image stats (surgical eye images):**
- Resolution: validation images `698×655` up to `1600×1600`; training frames mostly `1118×1120`,
  some `1920×1080`.
- Brightness (mean gray): ~24–116. Many images are dark (surgical microscope illumination).
- Contrast (gray std): ~25–69 (high contrast — dark background vs. lit eye).
- Near-white / glare fraction (`gray>235`): ≤0.11% — **few specular highlights** at image scale.
- Saturation (HSV): ~40–103; red-channel excess (suction ring red features) present.
- Format: JPEG (compression artifacts expected); colour (3-channel BGR).

**Iris-annulus stats** (using the cleaned limbus/pupil annotation geometry as ROI):
- Iris annulus pixel count: ~120k–800k px depending on image.
- Iris-region mean gray: ~17–70 (dark iris under surgical light).
- **Iris-region texture is weak**: Laplacian abs-mean mostly **1.2–3.3**; iris std mostly
  ~7.5–23. One image (eye_01) is higher (Laplacian ~10.8).
- Near-white fraction inside the annulus mostly **0** (few reflections), with 3 images ~0.9–1.1%.

**Interpretation (evidence-supported, conservative):** the only currently available eye images
are **dark surgical views with a relatively low-texture iris annulus**. This matters because
classical feature detectors (SIFT/ORB) and even learned detectors rely on measurable local
texture structure in the iris. Whether a given ELITA capture exhibits stronger, distinctive iris
structure **cannot be determined from the available data** (§21).

### 5.3 What is NOT known about ELITA (explicit)

The following cannot be answered from the repository and are **open, data-dependent questions**:
- Which specific images are pre-dock vs post-dock; pairing; same-eye correspondence.
- Image dimensions, resolution, colour/illumination of real ELITA captures.
- Iris/pupil/limbus visibility; reflection, occlusion, blur, compression, cropping.
- Whether iris appears circular/elliptical; geometry changes pre/post; metadata; annotations.

These are all pending **real ELITA data collection**.

---

## 6. Iris Feature Definition

### 6.1 Target anatomical feature classes

Evaluate each candidate for visibility, distinctiveness, repeatability, illumination/pupil-size/
blur/reflection sensitivity, and survival of pre/post positional change:

| Candidate feature | Anatomical basis | Expected utility for ELITA (from literature; to be validated on real data) |
|---|---|---|
| **Iris crypts** | Dark pits in the anterior iris stroma | Strong local contrast; distinctive; good candidates |
| **Furrows / contraction furrows** | Concentric grooves | Larger-scale; less stable under pupil size change |
| **Radial iris structures** | Radial fibres (esp. pigment ring / collarette area) | Moderate; may rotate multi-radially — need care |
| **Pigment patterns / pigment spots** | Localized melanin deposits | Distinctive and stable; good candidates |
| **Collarette / collarette-related structures** | Ring dividing pupillary and ciliary iris | **Reference landmark**; stability caution under pupil dilation |
| **Localized texture structures** | Small regions of distinctive texture | Usable if texture present |

### 6.2 Guiding principle

> Not every computer-vision keypoint is an anatomical iris feature. The goal is **stable,
> distinctive, repeatable** structure that survives pre/post positional changes and later
> matching.

Consequently, the design **rejects** using a raw keypoint detector alone as the final answer;
raw keypoints will be filtered by an explicit **feature-quality model** (§12) that enforces
anatomical plausibility and stability signals.

---

## 7. Iris ROI Strategy

- **Primary ROI** (uses existing geometry): an **annular (ring) region** bounded by the **limbus
  ellipse** (outer boundary) and the **pupil ellipse** (inner boundary), expressed in image-pixel
  coordinates.
- **Pupil exclusion**: pixels strictly inside the pupil ellipse are excluded.
- **Limbus / sclera boundary**: constrain to a fraction of the limbus radius (e.g. a fixed inset
  band, value **to be established**) to avoid sclera and limbus ambiguity.
- **Eyelids / eyelashes / occlusion**: use an occlusion mask (§8). Where present, eyelid occlusion
  removes that angular sector from usable feature area.
- **Reflections**: specular highlights are detected and masked out via the existing reflection
  utilities (brightness threshold + impainting) and/or red-highlight detection.
- **Poor-quality regions**: low-contrast or blurred regions are excluded by the feature-quality
  gate (§12).
- **Alternative ROI source**: the ML iris mask (`masks['iris']`) may veto/reinforce the
  geometric annulus.

**Recommendation:** an **annular ROI parameterized by pupil & limbus ellipses, eroded by an
occlusion/reflection mask** is appropriate. This reuses the reliable existing geometry and keeps
the region anatomically meaningful.

---

## 8. Imaging Disturbances

| Disturbance | Expected effect | Mitigation | Residual risk |
|---|---|---|---|
| Specular reflections | Fake keypoints/high-contrast patches | Reflection mask + inpainting (existing utilities) | Low, unless reflections are large/frequent (unknown for ELITA) |
| Lighting changes | Contrast/descriptor drift | CLAHE/normalization (existing) | Medium — descriptors drift across sessions |
| Blur / motion | Loss of fine texture | Quality gate rejects low-texture regions | Medium — may remove many features |
| Contrast changes | Threshold-sensitive detections | Normalization; use gradient-based measures | Low–Medium |
| Noise / compression (JPEG) | Spurious or unstable keypoints | Smoothing + quality gate; learned robustness | Low–Medium |
| Pupil dilation/constriction | Radial shifts; feature displacement relative to pupil | Use iris-relative (angular, normalized radial) coords; avoid pupil-proximal region | **Medium–High** — affects feature position stability |
| Eyelid occlusion | Sector missing; false boundaries | Occlusion mask; angular sector exclusion | Medium |
| Eyelash occlusion | Thin dark streaks → spurious keypoints | Masking + size/shape filtering | Medium |
| Partial iris visibility | Reduced usable area | Coverage/quality reporting (§12) | Medium |
| Pre/post imaging differences (camera alignment, field/position) | Scale/translation/rotation changes | Robust rotation-scale-invariant descriptors; iris-relative coords | **High** (core pre/post risk) |
| Post-dock (supine, ring-docked) changes | Pressure/geometry, eyelid, tear film changes | Revalidate on real post-dock data | **High — unknown** |

---

## 9. Candidate Feature-Detection Approaches

| Approach | Strengths | Weaknesses | Cost | Training/data | Robustness | Explainability | ELITA suitability | Pre/post matching | Integration | GPU? | ONNX-deploy? | Runs now? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. **Classical local features** (ORB/SIFT) | Simple, CPU, no training, ONNX-free | Weak on low-texture iris; not iris-specific; unstable density | Low | None | Low–Med | High | Low (given low measured iris texture) | Poor | Easy | No need | Yes (CPU) | Yes |
| 2. **Classical texture/gradient** (e.g. filtered patches, Gabor-like) | CPU, some iris structure | Manual channels; limited distinctiveness | Low | None | Med | High | Med | Med | Easy | No | Yes | Yes |
| 3. **Iris-specific representations** (1D/2D iris codes, Daugman-style Gabor/Log-Gabor filters) | Designed for iris structure; rotation-handling | Needs enrollment-style normalisation; circular coords assumption; may not fit surgical lighting | Med | Small / none | Med | Med | **Med** | **Med-High** | Med | No (CPU ok) | Possible | Yes |
| 4. **Learned feature extractors** (e.g. trained CNN embedding / keypoint net) | Strong on real data, invariant | Needs data+labels; GPU to train; ONNX possible; black-box | High | **Large** paired/feature data | High | Low | **High if data** | **High** | Med–High | Training yes; inference CPU possible | Yes (ONNX) | Training heavy |
| 5. **Hybrid (classical ROI + learned/classical descriptors)** | Balanced; reuse existing geometry; lower data need | More components | Med | Moderate | Med–High | Med | **High** | **High** | Med | Inference no | Yes | Yes |

### Recommended baseline for THIS project

Given (a) the **low measured iris texture** in the only available eye imagery, (b) **no ELITA
data yet**, and (c) the existing repository's CPU/ONNX-first philosophy, the recommended baseline
is a **hybrid approach (option 5)**:

1. **ROI**: annular iris region from existing pupil/limbus geometry + occlusion/reflection mask.
2. **Feature sampling on a regular iris-relative lattice** (angular × normalized radial), so
   features are reproducible in position and robust to translation/scale.
3. **Descriptor**: start with **classical, hand-defined local descriptors** (gradient histograms
   and/or a small set of log-Gabor/Gabor responses) that are rotation- and scale-aware, **no new
   dependency and no training required** — this de-risks the concept phase and runs on CPU.
4. **Quality gate**: retain only high-quality, stable features (§12).
5. Only if the proxy and/or early ELITA validation show classical descriptors insufficient for
   feature matching should a **learned embedding** be added (with the data/annotation plan of §11).

This recommendation is deliberately **not** "state-of-the-art for its own sake"; it is the minimal
approach consistent with the available data and existing architecture, and it explicitly defers
learned components until justified by evidence.

---

## 10. Recommended Concept Architecture

```
EXISTING ELITA/surgical pipeline
          |
          v
   [EXISTING] Pupil/limbus detection (UnifiedDetector)
          |
          v
   [NEW]   Iris ROI extraction (annular ROI from pupil/limbus ellipses)
          |
          v
   [NEW]   Image normalization (grayscale/CLAHE reuse; iris-relative sampling)
          |
          v
   [NEW]   Reflection / occlusion mask (reuse reflection utilities)
          |
          v
   [NEW]   Feature extraction (baseline: classical descriptors on lattice)
          |
          v
   [NEW]   Feature filtering (edge/anatomy filters)
          |
          v
   [NEW]   Quality / confidence per feature (feature-quality model)
          |
          v
   [NEW]   IrisFeatureSet (representation, §11)
          |
          v
   [FUTURE] Feature matching layer
          |
          v
   [FUTURE] Registration
          |
          v
   [FUTURE] Cyclotorsion θ
```

**Legend:** `EXISTING` = already present and reused; `NEW` = designed here, implemented in a later
phase; `FUTURE` = defined by interface here, not implemented.

**Scope guardrail:** this phase implements **nothing**. The `NEW` boxes are the target of the next
implementation phase; the `FUTURE` boxes must not be built in that phase.

---

## 11. Feature Representation

A candidate feature should carry:

| Attribute | Needed? | Notes |
|---|---|---|
| position (px) | **Yes** | Image-pixel coordinates (consistent with repo convention) |
| scale | Yes | Characteristic scale of the feature patch |
| orientation | Yes | Local orientation (for matching) |
| descriptor / embedding | **Yes** | The matching signature |
| iris-relative coordinates | **Yes** | (angular, normalized radial) — essential for pre/post correspondence |
| local contrast | Yes | Texture-strength proxy |
| visibility / occlusion flag | Yes | Whether the feature lies under an occlusion/reflection mask |
| image-quality proxy | Yes | Local blur/contrast/product of region |
| confidence | **Yes** | Composite quality score (§12) |

Stability-priority design: the system prefers **few high-quality stable features** over thousands
of unstable keypoints. Feature count is a reporting output, not a target to maximise.

### Future `IrisFeatureSet` contract (for the registration interface, §14)

```
IrisFeatureSet
  ├── source image / id
  ├── pupil & limbus ellipses (reference geometry)
  ├── list[IrisFeature]:
  │     ├── position (px)
  │     ├── scale
  │     ├── orientation
  │     ├── descriptor/embedding
  │     ├── iris_relative (angle, radial_norm)
  │     ├── local_contrast
  │     ├── visibility
  │     ├── image_quality
  │     └── confidence
  ├── ROI / occlusion metadata
  └── overall region coverage / usable-area fraction
```

---

## 12. Feature Quality / Confidence

- **Per-feature confidence**: composite of local contrast, texture/edgeness, visibility (not
  occluded/reflective), image-quality (blur/contrast), and descriptor self-consistency.
- **Quality gate / rejection**: drop features below a contrast threshold, inside the pupil or
  outside the eroded limbus, under occlusion/reflection, or overlapping eyelids/lashes.
- **Density control**: enforce a minimum angular separation and a target feature count band to
  ensure **spatial distribution** rather than clumping.
- **Region scoring**: report usable-iris fraction and the number/coverage of accepted features, so
  a sparse/occluded iris is flagged rather than silently under-featured.

The **confidence mechanism mirrors the existing `ConfidenceScorer` / `DetectionQuality` pattern**
(keep semantics consistent; do not invent a parallel, incompatible scale).

---

## 13. Pre/Post Stability Evaluation (future, not implemented)

The architecture must support measuring whether **A ↔ A′** is a reliable correspondence between
pre-dock and post-dock features. It will do so by preserving per-feature iris-relative coordinates,
descriptors, and confidences (so a future matcher can operate). Future metrics to implement
(later phase):

- Feature repeatability (fraction of features with a corresponding match)
- Localization error between matched features
- Descriptor similarity distribution
- Matching confidence / correct-match ratio
- Spatial distribution of matched features (avoid clumping)
- Number of reliable correspondences
- Failure rate (cases with too few/e.g. degenerate geometry)

These are **designed-for but NOT implemented** in this planning phase.

---

## 14. Future Register Interface

The feature detector must expose only the `IrisFeatureSet` (defined in §11) to a future
registration module — coordinates, descriptors, confidence, visibility, and quality metadata —
**without** exposing extraction internals. The registration module consumes that representation.
This keeps the detector and the matcher decoupled and testable independently.

---

## 15. Future Cyclotorsion Interface

The registration/matching stage (future) consumes:

```
PreDockFeatures + PostDockFeatures
        → Correspondences
        → Registration
        → Rotation θ (degrees, sign defined relative to astigmatism axis)
```

For accurate angular estimation the **feature-detection layer must preserve**:
- **iris-relative coordinates** (angular position is what maps to rotation),
- **descriptors** enabling unambiguous same-feature correspondence,
- **per-feature confidence** so the rotation estimate can be weighted/robust,
- **reference pupil/limbus geometry** for each set, and
- a **defined angular convention** (θ sign and reference axis) shared with the astigmatism-axis
  alignment support.

Nothing here is implemented in this phase; it is the contract the future detector and matcher
must honour.

---

## 16. Future Cyclotorsion Estimation (contract-level, not implemented)

Estimation of θ is a downstream phase. The interface preserves the data needed (§15). Angular
accuracy will depend on: descriptor distinctiveness, matching correctness, and the **number and
angular spread** of reliable correspondences — which is exactly why the feature layer reports
coverage and distribution rather than only feature count.

---

## 17. Proposed Repository Architecture

**Proposed future module location** (create in a later implementation phase, not now):

```
pupil_tracking/
    iris/                        # NEW (future)
        __init__.py
        roi.py                   # IrisROIExtractor: annular ROI from pupil/limbus + masks
        normalization.py         # iris-relative sampling / normalization
        features.py              # IrisFeature dataclass + IrisFeatureSet
        extraction.py            # baseline classical feature extraction
        filtering.py             # edge/anatomy/occlusion filtering
        quality.py               # per-feature quality/confidence, density control
        visualize.py             # overlay helpers for features (uses existing drawing patterns)
        detect.py                # top-level detect_iris_features(image, pupil, limbus, iris_mask)
    tests/
        test_iris_roi.py
        test_iris_extraction.py
        test_iris_quality.py
        test_iris_stability.py         # repeatability (future)
        test_iris_paired.py            # pre/post paired (future, needs paired data)
```

Responsibilities are kept modular; the module depends on existing `utils/types.py`
(`EllipseParams`), `preprocessing/` (reflection/normalization), and `core/` geometry. It **does
not** modify the existing detector, calibration, pupil, or limbus behaviour.

**GUI integration point (future):** an optional toggle to run iris-feature detection after
`UnifiedDetector.detect()` and draw features on the existing overlay; disabled by default to
preserve current behaviour.

**CLI/debug tooling (future):** a `scripts/extract_iris_features.py` debug script for offline
analysis on a directory of images (mirrors existing `scripts/debug_single_image.py` pattern).

**Visualization strategy:** extend the existing overlay/drawing utilities; features drawn as
points with orientation tick + confidence colour scale.

---

## 18. Model / Data Requirements

### First concept model — recommendation

- **Start classical (no learned model)**: the baseline of §9 (hybrid classical ROI + classical
  descriptors) requires **no training data, no labels, no GPU**, and runs on the current CPU-only
  system. This directly answers the concept question "can useful iris features be identified from
  real ELITA images?" with the least risk.
- **Learn only if needed**: escalate to a fine-tuned/self-supervised learned extractor **only if**
  the classical baseline fails pre/post matching on real data. That escalation requires a plan
  below.

### Requirements for the learned escalation (future, only if classical is insufficient)

| Item | Requirement |
|---|---|
| Paired pre/post ELITA images (same eye) | **Required** — feature correspondence/rotation ground truth |
| Feature-level annotations | Preferred (matched feature pairs) for supervised matching |
| Minimum useful dataset | **To be established** after proxy validation; likely dozens+ of eyes |
| Validation dataset | Held-out eyes (image-level split to prevent leakage, as the existing `split_by_images` does) |
| Test dataset | Independent eyes, incl. varied illumination/occlusion |
| Augmentation | Rotation, scale, illumination, blur, reflection synthesis (after core validated) |
| Annotation budget | Keep small; prefer weak/self-supervised correspondence if possible |

Data collection / annotation is **explicitly out of scope** for this planning phase.

---

## 19. Testing Strategy (design for the implementation phase)

1. **Unit tests**: ROI construction, iris-relative coordinate mapping, filtering, quality math.
2. **Image-level tests**: detection runs on real surgical images; produces valid `IrisFeatureSet`.
3. **Repeatability tests**: same image processed repeatedly → stable feature set (future metric).
4. **Pre/post paired-image tests**: only possible once paired ELITA data exists.
5. **Disturbance robustness tests**: synthetic perturbation (blur, illumination, rotation, scale)
   to measure feature stability.
6. **Regression tests**: existing 243-test suite stays green (baseline 242/243); new module must
   not change detector/calibration/pupil/limbus outputs.
7. **Performance tests**: runtime budget (e.g. target **to be established** ms/image on CPU).
8. **Visualization / manual validation**: overlay output reviewed on real images.

**Success/failure definition** per test type is explicit in §20 and kept threshold-free where data
does not justify a number.

---

## 20. Success Criteria (first concept model)

Acceptance must be demonstrated on **real images**; where the data does not justify a hard number,
the criterion is marked **to be established experimentally**.

1. Iris region isolated reliably (annular ROI validated on available surgical images).
2. Features detected on real eye images (not synthetic only).
3. Features not dominated by reflections (reflection mask excludes glare-dominated areas).
4. Features sufficiently distributed across usable iris regions (angular spread reported).
5. Features carry meaningful quality/confidence information (rank correlates with texture strength).
6. Features can be repeated across suitable pre/post images — **cannot be verified without ELITA
   paired data**; the correct proxy first step is **repeatability on the same/undisturbed image and
   under synthetic perturbation** (thresholds to be established).
7. Output is visualizable via existing overlay patterns.
8. Failure cases identifiable (explicit low-coverage / low-quality reporting).
9. Runtime practical on the existing machine (CPU; budget to be established).

All numeric thresholds (feature count band, contrast threshold, angular separation, runtime) are
**to be established experimentally** once real data is available.

---

## 21. Risks

- **Data absence (highest):** no ELITA or paired pre/post-dock images in the repository. All
  ELITA-feasibility claims are inherently unverifiable until data is provided. Mitigation:
  validate the concept pipeline on existing surgical imagery as a proxy, then re-validate on ELITA.
- **Weak iris texture** in the only available images: classical feature detectors may return few
  stable features. Mitigation: quality-gated, iris-relative lattice + report coverage.
- **Pupil size change between pre/post:** radial feature displacement; mitigated by iris-relative
  coordinates and avoiding the pupil-proximal band.
- **Pre/post imaging differences** (scale/translation/rotation, illumination, occlusion, tear
  film): the core matching risk; only resolvable with real paired data.
- **Stale baseline test:** the existing 1 failing test is a documented pre-existing issue (old
  hardcoded expectations); not a regression and not touched here.
- **Dirty working tree in this checkout:** many unrelated uncommitted changes exist; this phase
  must not and does not touch them (staging discipline, §Git).
- **GT / annotation consistency:** prior phase documents reference a `corrected_annotations`
  file that is **absent from this checkout**, and on-disk `clean/annotations` GT disagrees with the
  documented Phase-16 table for 8/13 images. This does not block the *iris* plan but signals that
  annotation provenance should be re-verified before any future clinical validation uses it.

---

## 22. Open Questions

**Known facts**
- Existing pupil/limbus pipeline, ONNX production inference, and dataclass result structures are
  present and functional.
- Only surgical eye imagery (no ELITA, no paired pre/post) exists in the repository.
- Measured iris-region texture in the available images is low (Laplacian abs-mean mostly 1.2–3.3).

**Evidence-supported conclusions**
- A hybrid classical-ROI + classical-descriptor baseline is low-risk, dependency-free, runs on the
  current CPU/ONNX-first system, and aligns with the existing architecture.
- The existing pupil/limbus geometry + reflection utilities are directly reusable for an annular
  iris ROI.
- ELITA-specific feasibility cannot be established from the available data.

**Engineering assumptions (to be confirmed by experiment)**
- An annular ROI parameterized by pupil/limbus ellipses + occlusion/reflection mask is the correct
  usable-iris region.
- Iris-relative (angular / normalized-radial) coordinates are the right invariant representation.
- A classical descriptor baseline is sufficient for a first concept validation.

**Open questions requiring experimental answers**
1. Do real ELITA pre/post images have enough iris texture for reliable feature detection?
2. Are pre/post captures paired and same-eye? What are their geometry/illumination/occlusion?
3. How much pupil-size change and positional variation occurs between pre and post dock?
4. Is a learned extractor necessary for reliable pre/post matching, or is classical sufficient?
5. What are justified numeric thresholds (feature count, contrast, runtime, match confidence)?

---

## 23. Recommended Implementation Phases

- **Phase I (next, implementation of `NEW` boxes in §10):** `pupil_tracking/iris/` module with ROI
  extraction, normalization, baseline classical feature extraction, filtering, quality, and
  visualization, integrated as an **disabled-by-default** extension. Regression-safe (existing
  suite stays green). Validated on the existing surgical imagery (proxy) + synthetic perturbation.
- **Phase II:** repeatability + disturbance-robustness metrics on the proxy data; tune quality
  gate and coverage reporting; set initial numeric thresholds.
- **Phase III:** **requires real paired ELITA data** — pre/post paired testing, feature
  correspondence, and stability metrics.
- **Phase IV (future):** feature matching and registration.
- **Phase V (future):** cyclotorsion θ estimation and astigmatism-axis alignment support.

Only Phase I belongs to the immediate next development phase; implementation is **out of scope** of
this planning document.

---

## 24. Final Recommendation

Proceed to implement the iris-feature-detection **concept model** as a hybrid classical approach
(§9, §10) on the existing architecture, in a **new, modular `pupil_tracking/iris/` package** that
reuses existing pupil/limbus geometry, reflection utilities, confidence patterns, and visualization
tools. Validate on existing surgical imagery as a proxy and under synthetic perturbation first;
obtain and re-validate on **real paired ELITA pre/post-dock images** before any feature-matching or
registration work.

**The immediate goal is iris feature detection from ELITA pre-dock and post-dock images.
Feature matching, registration and cyclotorsion estimation are subsequent phases.**

Nothing in this report has been implemented; no production code, model, calibration, pupil, or
limbus behaviour has been changed.

---

### Appendix — verification performed for this document

- Read `opencode.md` and `gemini.md` in full.
- Confirmed **no `files/` directory** exists (requirement docs unavailable).
- Deep read-only inspection of the existing pipeline via exploration agent (detector, types,
  preprocessing, ML/ONNX/PyTorch/FastInference, video, GUI, tests, annotations, requirements).
- Quantitative image analysis of the available surgical images (whole-image and iris-annulus
  statistics) — see §5.2.
- Confirmed via repository-wide search that **no ELITA, iris-feature, cyclotorsion, or
  registration implementation** exists.
- This is a documentation-only change; no tests are affected. Baseline test state is the documented
  **242/243 (1 pre-existing failure)**.

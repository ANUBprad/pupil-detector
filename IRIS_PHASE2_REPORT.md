# Phase II Evaluation Report — Iris Feature Repeatability & Robustness

> **Phase**: II (iris feature repeatability / robustness / distribution)
> **Date**: 2026-08-29
> **Branch**: `main` (target → https://github.com/ANUBprad/pupil-detector.git)
> **Prior**: Phase I verdict = READY (`9bb8c81`)
> **Scope**: Clinical proxy images + controlled synthetic perturbations ONLY
> **Not in scope (NOT implemented)**: feature matching, registration,
>   cyclotorsion/rotation estimation, astigmatism correction, pre/post-dock
>   matching, real ELITA data.

---

## 0. Executive verdict

**NEEDS IMPROVEMENT.**

The Phase I detector is **deterministic**, **translation-robust**, and its
masking/occlusion handling is **correct**. But it is **not yet robust to the
perturbations most relevant to the ELITA use case** — rotation, scale and noise —
and on most clinical proxy images it produces **too few / too angularly-clustered
features** to support reliable correspondence. A real ELITA pre/post dock pair
differs by rotation (cyclotorsion), scale (dock magnification) and illumination;
under mild versions of exactly those changes the detector loses 40–60 % of its
features. It is **not ready for real ELITA data yet**.

---

## 1. Method (summary)

Evaluation-only harness:

* `pupil_tracking/iris/robustness.py` — pure, deterministic evaluation library
  (repeatability, spatial distribution, inverse geometric mapping, perturbations).
* `scripts/iris_phase2_eval.py` — runner over the 12 clinical proxy images.
* `pupil_tracking/tests/test_iris_robustness.py` — 19 deterministic unit tests.

**Repeatability model** (controlled-correspondence): baseline features are
extracted on the unperturbed image with fixed geometry; the image is perturbed and
re-extracted; two features "correspond" if they fall within `ang_tol = 3°` and
`rad_tol = 0.06` (halved lattice spacings) of each other in the **baseline
normalised iris coordinate frame**. Repeatability rate = fraction of baseline
features with a corresponding match (plan §13). The inverse geometric mapping
used for translate/rotate/scale was **numerically verified to be exact**
(0 px / 0° residual on synthetic round-trips) — see §6 (verification).

---

## 2. Dataset / coverage

12 clinical proxy images (`clinical_data/clean/*.jpeg`). **5/12 yielded a valid
iris ROI** (eye_01, eye_02, eye_03, eye_11, eye_13) — the same set as the Phase I
smoke baseline. 7 images (eye_06..eye_10, eye_12, eye_14) produce no valid pupil
/ limbus geometry under the production `UnifiedDetector`, so no iris features can
be measured there. n=5 images, and two of them (eye_11: 3 features, eye_13:
9 features) are **very sparse** — see §8 (representativeness).

---

## 3. Baseline feature statistics (%A) and spatial distribution (%B)

FACTUAL OBSERVATION: per valid image

| image | limbus_r px | accepted | usbl | angCov | radCov | conc | ent | quark quadrants | medianQual | ms |
|-------|-------------|----------|------|--------|--------|------|-----|------------------|------------|-----|
| eye_01 | 225.9 | 72 | 0.70 | 1.00 | 0.75 | 0.40 | 1.00 | 4 | 0.674 | ~115 |
| eye_02 | 214.4 | 23 | 0.72 | 0.67 | 0.75 | 0.22 | 0.78 | 4 | 0.358 | ~117 |
| eye_03 | 219.1 | 17 | 0.72 | 0.58 | 0.75 | 0.24 | 0.69 | 4 | 0.362 | ~112 |
| eye_11 | 357.5 | 3 | 0.69 | 0.25 | 0.75 | 0.33 | 0.44 | 2 | 0.390 | ~132 |
| eye_13 | 243.8 | 9 | 0.72 | 0.17 | 0.50 | 0.44 | 0.28 | 2 | 0.352 | ~123 |

INTERPRETATION:
* Only eye_01 (the same image that dominates phase-history analysis) has a robust,
  well-distributed feature set (72 features, full angular coverage).
* 3 of 5 valid images (eye_03/11/13) have **fewer than 20 features** and
  **angularly clustered** coverage (≤ 0.58 angular coverage, ≤ 2 quadrants).
* Plan §16 ties cyclotorsion accuracy to the *number and angular spread of
  reliable correspondences*; 3–17 features clustered in 2 quadrants is inadequate.
* Median confidence is low (0.35–0.39) on 4 of 5 images — the detector is not
  finding many *strong* texture features, consistent with the low measured iris
  texture on the available proxy imagery (plan §5).

---

## 4. Determinism (%C)

FACTUAL OBSERVATION: all 5 valid ROIs are **fully deterministic** across 5
repeated identical runs — identical accepted count and identical normalised
coordinates each time.

INTERPRETATION: no source of randomness in the detector; results are
reproducible. Good.

---

## 5. Perturbation robustness (%D) — repeatability in the normalised frame

FACTUAL OBSERVATION: aggregate repeatability across the 5 valid images
(mean over the perturbation's two strengths):

| perturbation | rep_mean | rep_min | retained_mean | median angGap(°) |
|--------------|----------|---------|---------------|------------------|
| brightness   | 0.877 | 0.667 | 0.955 | 0.0 |
| contrast     | 0.787 | 0.348 | 1.357 | 0.0 |
| gamma        | 0.887 | 0.667 | 1.159 | 0.0 |
| noise (σ2,6) | 0.575 | 0.217 | **8.073** | 0.0 |
| blur (3,7)   | 0.593 | 0.174 | 0.785 | 0.0 |
| sharpen      | 0.934 | 0.783 | 1.799 | 0.0 |
| translate    | 1.000 | 1.000 | 1.000 | 0.0 |
| rotate (3°)  | 0.402 | **0.000** | 0.830 | 1.75 |
| scale (3%)   | 0.409 | **0.000** | 0.713 | 0.88 |

Per-image mean repeatability matrix:

```
image             bright contra gamma noise blur sharp transl rotate scale
eye_01 (72 feats)  1.00  0.99  0.99  0.71  0.76  0.95  1.00  0.61  0.56
eye_02 (23 feats)  0.89  0.63  0.87  0.41  0.26  0.89  1.00  0.24  0.24
eye_03 (17 feats)  0.88  0.71  0.79  0.53  0.56  0.88  1.00  0.38  0.53
eye_11 ( 3 feats)  0.83  0.83  0.83  0.83  0.50  1.00  1.00  0.00  0.17
eye_13 ( 9 feats)  0.78  0.78  0.94  0.39  0.89  0.94  1.00  0.78  0.56
```

INTERPRETATION:
* **Translation is perfectly robust (1.00)**: the iris-relative normalised
  coordinate frame is translation-invariant. Expected and confirmed.
* **Photometric changes are handled well** (brightness/gamma/sharpen 0.88–0.93),
  moderate for contrast (0.79).
* **Noise and blur degrade the detector** to ~0.55–0.59 repeatability, and noise
  creates ~8× *spurious* features (`retained_mean` 8.07) — the raw intensity-texture
  response is not noise-robust.
* **Rotation and scale are the critical weakness**: a mild 3° rotation or 3% scale
  destroys ~60 % of features, and on some images (eye_11 rotation, eye_02) it is
  0.24–0.00. The inverse geometric mapping is exact (verified), so this is a real
  detector-level **content** change: rotating/scaling the image changes *which*
  features are accepted, not merely their coordinates. The intensity-texture
  detector is not rotation/scale invariant.
* Real pre/post-dock ELITA pairs differ by rotation and scale; these are exactly
  the worst cases measured. This is the primary blocker for a "READY" verdict.

---

## 6. Harness verification (why the numbers are trustworthy)

To actively disprove a false negative, the geometric correspondence was verified
numerically: transforming a baseline iris point by translate/rotate/scale and then
mapping it back recovered the original pixel and normalised coordinates to **0 px /
0° residual** on synthetic round-trips (probe in repo temp, plus 3 unit tests).
During this work a genuine bug in the harness's inverse-scale mapping was found and
fixed (the y-axis omitted the `/factor` division, giving up to 13 px error); this
had **falsely** depressed the pre-fix `scale` repeatability from 0.409 → 0.254
(direction of bias: harness error, not detector error). After the fix the mapping
is exact and the reported numbers are the true detector figures. Unit tests
(`test_map_point_back_*`) lock this in so the report is reproducible.

---

## 7. Occlusion / reflection robustness (%E)

FACTUAL OBSERVATION (geometry-aware annulus occluder, disc ~45 % of annulus width
placed mid-annulus along a random ray; external_occlusion mask passed to detector):

| image | occ_r px | accepted | acc→ | usbl | usbl→ | inMask | rep |
|-------|----------|----------|------|------|-------|--------|-----|
| eye_01 | 64 | 72 | 63 | 0.701 | 0.628 | 0 | 0.875 |
| eye_02 | 69 | 23 | 19 | 0.718 | 0.630 | 0 | 0.826 |
| eye_03 | 71 | 17 | 17 | 0.718 | 0.630 | 0 | 1.000 |
| eye_11 | 99 | 3 | 3 | 0.693 | 0.621 | 0 | 1.000 |
| eye_13 | 78 | 9 | 9 | 0.720 | 0.633 | 0 | 1.000 |

INTERPRETATION:
* **`inMask = 0` on every image** — no accepted feature ever lands inside the
  occluded disc. The detector **correctly honours** the external occlusion mask
  (masking.py "usable = annulus ∩ ¬reflection ∩ ¬external occlusion").
* usable_fraction drops as expected and the surviving features are the un-occluded
  ones (rep high on eye_01/eye_02 where the disc removes features).
* On very sparse images (eye_03/11/13) the disc happened not to coincide with any
  accepted feature, so acc/rep are unchanged — this is honest (nothing lost), not
  a masking failure. Reflection and occlusion handling is **correct**.

---

## 8. Threshold sensitivity (%F)

FACTUAL OBSERVATION: sweeping `min_contrast` (mean candidates then mean accepted
over the 5 valid images):

```
min_contrast  0.0  1.0  2.0  4.0  6.0  8.0
mean cand     565  521  288   60   48   41
mean acc       72   72   52   25   21   18
max  acc       72   72   72   72   72   72
```

INTERPRETATION:
* The contrast gate genuinely works — `num_candidates` (count passing the gate
  before angular pruning) falls 565 → 41 across the sweep.
* **On eye_01 the accepted count stays 72 for every threshold up to 8.0**: all its
  candidates have response ≥ 8, so contrast never binds there; the accepted count
  is set entirely by the angular-separation / max-features **density pruning**.
* On the other images accepted count is contrast-sensitive (72→52→25 as the gate
  tightens at the pooled level).
* **No defect**: `.min_contrast = 4.0` sits in the sensible range; the flatness on
  eye_01 is a density-limited regime, not a broken knob. Per the plan §12, density
  control is intentional. **No threshold changed** in this phase (no concrete
  defect justified it).

---

## 9. Quality vs stability (%G)

FACTUAL OBSERVATION: Spearman rank correlation between baseline per-feature
confidence and retention under photometric perturbations, defined-only aggregation
(n=20 image×perturb samples): **mean = 0.22, min = −0.14, max = 0.66**.

INTERPRETATION:
* The per-feature confidence is only a **weak, inconsistent** predictor of whether
  a feature survives a perturbation. On some samples higher-confidence features
  are more stable (positive), on others the relationship is ~0 or even slightly
  negative.
* Implication: today `confidence` should **not** be used as a hard selection
  criterion for correspondence robustness. The plan §12's composite-confidence
  goal (contrast + texture + boundary + visibility + self-consistency) is not yet
  delivering a strongly stability-correlated score on this proxy imagery.

---

## 10. Performance (%H)

FACTUAL OBSERVATION: iris-only (mask build + extraction) timing on the 5 valid
images: **mean ≈ 116 ms, median ≈ 114 ms, worst ≈ 124 ms**.

INTERPRETATION: negligible vs. the ~0.5–2 s/image full-pipeline detection cost
(document measures unified detection at 483 ms–1.3 s). Iris feature extraction is
not a performance concern.

---

## 11. Production safety (%I)

FACTUAL OBSERVATION: the iris package remains **importable and separately
invocable**; it is **not** imported by `pupil_tracking.core.UnifiedDetector`,
`launch_gui.py`, or any production path (confirmed in Phase I, unchanged here). The
Phase II additions (`robustness.py`, `test_iris_robustness.py`,
`scripts/iris_phase2_eval.py`) are **evaluation-only**: nothing is wired into the
production pipeline, no matching, registration, cyclotorsion, or astigmatism logic
was added, and no production model / clinical data / thresholds were modified.

---

## 12. Reproducibility & test status

* New unit tests: **19 passed** (`test_iris_robustness.py`) — deterministic,
  synthetic, non-tautological.
* Full suite (excluding the documented broken untracked WIP
  `test_runtime_profile.py`, which fails at import on a missing symbol): **296
  passed / 14 skipped / 7 failed**. The 7 failures are exactly the documented
  pre-existing set (1× `test_corrected_output_in_help` — dirty-tree whitespace;
  5× `test_modular_calibration` — dirty-tree WIP; 1×
  `test_eye_01_unchanged_after_ring_constraint` — stale test). **No new failures
  introduced by Phase II** (prior Phase I suite: 277 passed; delta +19 = the new
  robustness tests).
* All evaluation numbers above were produced by the committed, deterministic
  runner (`scripts/iris_phase2_eval.py`) against committed reproducible logic.

---

## 13. Three-way classification

### FACTUAL OBSERVATIONS (measured, reproducible)
1. Deterministic: identical repeated runs yield identical feature sets.
2. Translation-invariant normalised coordinates → perfect translation robustness.
3. Occlusion/reflection masking is correct (no feature ever in the occluded disc;
   usable_fraction drops as expected).
4. `min_contrast` gate works (candidates 565→41 over the sweep); accepted count is
   density-limited (not contrast-limited) on eye_01 across thresholds 0–8.
5. Rotate (3°) rep ≈ 0.40; scale (3%) rep ≈ 0.41; noise rep ≈ 0.58 / ~8× spurious
   features; blur rep ≈ 0.59. Inverse geometric mapping verified exact (0 px).
6. 5/12 images yield a valid ROI; 3 of those 5 have ≤ 17 accepted features and
   ≤ 2-quadrant angular coverage.
7. Iris-only timing ≈ 116 ms mean; iris package not wired into production.
8. Full suite: 296 passed / 14 skipped / 7 failed (all 7 pre-existing).

### INTERPRETATIONS (the above, in context)
- The detector is correctness-sound and deterministic, and its geometry/mask
  primitives are reliable.
- But it is a **non-invariant intensity-texture detector**: acceptable under
  photometric change and translation, **not** under rotation/scale/noise — the
  very changes a real pre/post-dock ELITA pair will exhibit.
- Feature counts and angular spread are marginal-to-insufficient for correspondence
  on most proxy images (3–23 features; ≤ 2 quadrants on 3 images).
- Per-feature `confidence` is a weak stability predictor today.

### RECOMMENDATIONS (future work, NOT done here)
1. **Rotation/scale invariance is the priority**: switch to (or add) rotation/
   scale-invariant content (e.g. orientation-aligned patches, local affine/steerable
   filters, or gradient-histogram descriptors) and re-measure rotate/scale
   repeatability before any real-ELITA matching attempt.
2. **Noise robustness**: add pre-extraction denoising / stronger quality gating so
   noise does not multiply spurious features 8×.
3. **Feature density/coverage**: relax the angular-suppression only where texture
   is dense, or increase angular sub-division, so sparse-iris images (eye_11/13)
   yield a larger, better-spread feature set; audit whether the ≤17-feature cases
   are a data-realistic limit of low iris texture (plan §5) versus detector policy.
4. **Re-evaluate confidence scoring** if it is intended to gate correspondence;
   today it does not strongly separate stable from unstable features.
5. **Re-run this harness unchanged on the first real paired ELITA captures**; the
   numbers here are proxy-image-bound and may differ on real surgical imaging.

---

## 14. Final verdict

**NEEDS IMPROVEMENT — NOT "READY FOR REAL ELITA DATA".**

Rationale (deliberate, evidence-based): the Phase I detector is deterministic,
geometry/masking-correct, and photometric-and-translation robust — but it is not
robust to rotation, scale or noise, and it yields too few / too clustered features
for reliable correspondence on most of the only available clinical proxy images.
Rotation and scale are intrinsic to the pre/post-dock ELITA use case, so shipping
the current feature layer directly into correspondence would risk a high
mismatch/failure rate. The blocker is a **content-invariance** and **feature-count/
coverage** shortfall, not a correctness bug; it is addressable in a future phase
(§13 recs), and this harness provides the exact, reproducible baseline against
which an improved detector must be judged.

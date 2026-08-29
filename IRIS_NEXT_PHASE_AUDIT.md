# Iris Feature Detection — Phase III Independent Audit & Next-Phase Recommendation

> **Audit**: post-Phase-III independent verification (actively attempting to disprove conclusions).
> **Date**: 2026-08-29
> **Repository**: https://github.com/ANUBprad/pupil-detector (branch `main` = `target`)
> **HEAD (before audit)**: `5ac3ba4` "feat(iris): add Phase II repeatability & robustness evaluation harness"
> **Status**: Phase III verified; next phase recommended. **Phase III code remains uncommitted** in the working tree — this audit does not commit or modify it, and commits only this document.

---

## 0. Executive summary

1. **Phase III (smoothed-Sobel gradient response, response-dominant confidence) is real and verified.** In a controlled A/B run (identical committed harness, identical production model, identical 5-image clinical-proxy set), Phase III improves the four ELITA-critical perturbations by **+0.21 mean repeatability** (0.495 → 0.705): noise 0.575→0.938, blur 0.593→0.840, rotation 0.402→0.502, scale 0.409→0.538. It also **eliminates the catastrophic noise-feature flood** (retention 8.07× → 1.62×), doubles throughput (~386 ms → ~193 ms mean), and materially improves feature counts and angular distribution on the previously sparse images (eye_11: 3→25 accepted, eye_13: 9→20). The 37-test iris suite passes; the two full-suite failures are **pre-existing and unrelated to Phase III**.

2. **But Phase III is NOT finished or committed.** No `IRIS_PHASE3_REPORT.md`, no Phase III eval script, no Phase III commit exist. Phase III lives only as 3 uncommitted file edits (+1 test file). This audit's measured BEFORE/AFTER table (§4) is the missing evidence.

3. **Rotation and scale remain the dominant weaknesses** (≈0.50–0.54 mean repeatability; eye_11 worst at 0.18/0.22). This is structural: features are snapped to a fixed 5° angular lattice and matched with a fixed 3° tolerance, so correspondence under rotation is inherently under-measured.

4. **The single largest coverage bottleneck is upstream**: the production `UnifiedDetector` does not produce a **limbus ellipse on 7/12 images** (eye_06,07,08,09,10,12,14), so the iris layer cannot even evaluate there. The iris module does not cause this, but it inherits it unconditionally.

5. **Real ELITA paired data still does not exist in the repository** (independently re-verified). The scientifically correct next phase is therefore **not** "matching on real ELITA pairs" — that stays blocked on data. The correct next phase is **correspondence & rotation-recovery capability on synthetic proxy pairs**, delivered as a ready-to-run evaluation layer that (a) proves the descriptor layer can uniquely and accurately recover rotation/scale/radial correspondence, and (b) is unchanged when the first real ELITA pair arrives.

---

## 1. Audit mandate and method (how this was verified)

The audit independently re-derived every claim from a clean checkout, without trusting prior agent reports, and attempted to DISPROVE the "Phase III works" conclusion.

| Step | Action | Result |
|------|--------|--------|
| 1 | Verify Phase III completeness (git log, git status, glob) | **Phase III uncommitted; no report; no eval script.** |
| 2 | Read `opencode.md`, `gemini.md`, plan, Phase II report, all `pupil_tracking/iris/*` sources (committed + working tree) | Documented intent understood. |
| 3 | Confirm `files/` requirements directory | Does **not** exist (same as plan found). |
| 4 | Verify production model artifacts | Hashes **differ** from CLAUDE.md and from `manifest.json` (stale) — see §8. |
| 5 | Build two clean git worktrees at `5ac3ba4`: `phase2` (pure baseline) and `phase3` (= baseline + the 3 Phase III iris diffs) | Both importable; `git diff --stat` in `phase3` shows exactly the 3 iris files changed. |
| 6 | Copy the identical gitignored ONNX model + clinical images into both worktrees | Same `segmentation_quantized.onnx` used by both runs (confirmed in both logs). |
| 7 | Run `scripts/iris_phase2_eval.py` in each worktree | BEFORE/AFTER metrics captured (§4). |
| 8 | Run iris test suites in `phase3` | 37/37 pass (34 prior + 3 new Phase III tests). |
| 9 | Run full suite in `phase3` | 300 passed, 14 skipped, **2 failed**; both failures also fail in the dirty main tree and are unrelated to iris (see §7). |
| 10 | Instrument ROI validity on all 12 images | Confirmed the 7 invalid images all fail because **limbus is not detected** (pupil is). |
| 11 | Search repository for real ELITA / paired / pre-post data | **None** (only the 12 surgical proxy images). |

**Verification that control was clean:** the same detector output was reproduced in both worktrees (pupil radius, limbus radius identical per image), so the measured delta is attributable only to the 3 Phase III iris edits. The Phase II run also **reproduces the Phase II report's numbers exactly** (noise repeatability 0.575, retention 8.073, rotation 0.402, scale 0.409), confirming the harness and baseline are stable.

---

## 2. What Phase III changed (the independent diff summary)

Committed baseline (`5ac3ba4`) → working tree (only `pupil_tracking/iris/config.py`, `detect.py`, `extraction.py`, plus tests):

1. **Response operator**: mean-absolute-**Laplacian** of an unsmoothed patch → **mean Sobel gradient magnitude** of a pre-smoothed patch (`smooth_sigma = 1.0`), sampled from a box-filtered gradient map (per-candidate cost ≈ 0).
2. **Confidence**: `0.5·resp + 0.3·contrast + 0.2·clearance` → `0.7·resp + 0.3·clearance` (response-dominant; contrast dropped for ~zero predictive value).
3. **Gate recalibration**: `min_contrast` 4.0 → 8.0 so accepted-feature density stays comparable (improvement is real, not count inflation).
4. Three new deterministic tests (config default, noise-count stability `n_noisy ≤ n_clean·1.6 + 2`, confidence monotonicity/boundedness).

Independent characterisation: the changes are **sound and directionally exactly where Phase II said they were needed** (noise/blur). The Sobel + pre-smoothing is a first-order operator and fundamentally less noise-amplifying than a raw second derivative — the mechanism claimed matches the observed effect.

---

## 3. Controlled BEFORE/AFTER measurement

Same harness (`scripts/iris_phase2_eval.py`, committed at `5ac3ba4`), same model, same images.

### 3.1 Baseline feature statistics (valid ROI images; Phase III detector metrics)

| image | limbus_r px | accepted II→III | usable frac II→III | angCov II→III | radCov II→III | quadrants II→III | median conf II→III |
|-------|------------|-----------------|--------------------|---------------|---------------|------------------|---------------------|
| eye_01 | 225.9 | 72 → 72 (capped) | 0.70 → 0.70 | 1.00 → 1.00 | 0.75 → 0.50 | 4 → 4 | 0.674 → 0.900 |
| eye_02 | 214.4 | 23 → 26 | 0.72 → 0.72 | 0.67 → 0.75 | 0.75 → 1.00 | 4 → 4 | 0.358 → 0.900 |
| eye_03 | 219.1 | 17 → 22 | 0.72 → 0.72 | 0.58 → 0.67 | 0.75 → 0.75 | 4 → 4 | 0.362 → 0.961 |
| eye_11 | 357.5 | **3 → 25** | 0.69 → 0.69 | 0.25 → 0.83 | 0.75 → 1.00 | **2 → 4** | 0.390 → 0.575 |
| eye_13 | 243.8 | **9 → 20** | 0.72 → 0.72 | 0.17 → 0.42 | 0.50 → 0.75 | 2 → 3 | 0.352 → 0.900 |

Candidates total 298 → 447; accepted total 124 → 165. The Phase II report's "too few / too angularly-clustered" blocker on 3/5 images is **materially resolved** (eye_11 and eye_13 now exceed 20 features with multi-quadrant coverage).

### 3.2 Perturbation repeatability (mean across the 5 valid images)

| perturbation | Phase II | Phase III | Δ | interpretation |
|--------------|----------|-----------|-----|----------------|
| brightness (±25) | 0.877 | 0.902 | +0.025 | robust, unchanged |
| contrast (0.8/1.2×) | 0.787 | 0.812 | +0.025 | moderate, ~unchanged |
| gamma (0.8/1.2) | 0.887 | 0.854 | −0.033 | very minor drift (see §6) |
| **noise (σ=2,6)** | **0.575** | **0.938** | **+0.363** | primary Phase II weakness fixed |
| **blur (k=3,7)** | **0.593** | **0.840** | **+0.247** | primary Phase II weakness fixed |
| sharpen (0.2/0.6) | 0.934 | 0.972 | +0.038 | robust |
| translate (4 px) | 1.000 | 1.000 | 0.000 | perfectly translation-invariant |
| **rotate (±3°)** | **0.402** | **0.502** | **+0.100** | improved but still weak |
| **scale (0.97/1.03)** | **0.409** | **0.538** | **+0.129** | improved but still weak |
| **Mean, all 9** | **0.718** | **0.818** | **+0.100** | — |
| **Mean, ELITA-critical (noise+blur+rotate+scale)** | **0.495** | **0.705** | **+0.210** | — |

Worst-case (per-image minimum) repeatability — Phase III **no longer collapses to 0**:

| perturbation | Phase II min | Phase III min |
|--------------|--------------|---------------|
| noise | 0.217 | 0.640 |
| blur | 0.174 | 0.360 |
| rotate | **0.000** | 0.120 |
| scale | **0.000** | 0.200 |

### 3.3 Retention (perturbed count / base count) — spurious-feature control

| perturbation | Phase II | Phase III |
|--------------|----------|-----------|
| noise | **8.073×** (spurious flood) | **1.620×** |
| contrast | 1.357 | 1.094 |
| sharpen | 1.799 | 1.101 |
| gamma / brightness / blur / translate | ~0.8–1.2 | 0.89–1.07 |

Phase II's signature pathology (noise injecting ~8× spurious features) is eliminated.

### 3.4 Other harness sections

| Section | Phase II | Phase III |
|---------|----------|-----------|
| Determinism (5× repeat) | all True | all True |
| Occlusion / occlusion-annulus repeatability | eye_01 .875, eye_02 .826, eye_03 1.000, eye_11 1.000, eye_13 1.000 | eye_01 .875, **eye_02 .769**, **eye_03 .773**, eye_11 1.000, **eye_13 .650** |
| Threshold sensitivity (default settings) | min_contrast 4.0 → mean_cand 59.6, mean_acc 24.8 | min_contrast 8.0 → mean_cand 89.4, mean_acc 33.0 |
| Quality ↔ retention Spearman | mean 0.220, min −0.142, max 0.661 (n=20) | mean 0.270, min −0.400, max 0.636 (n=18) |
| Iris-only runtime (mean / median / worst) | 386.1 / 303.7 / 814.3 ms | **193.3 / 190.1 / 239.8 ms** |
| Production safety (importable) | True | True |

### 3.5 Verdict on Phase III

**Phase III materially improves robustness in exactly the two areas Phase II flagged as blockers (noise, blur), halves the runtime, and lifts feature quantity/distribution on the sparse images.** The two residual weaknesses (rotation, scale) improved but are not resolved, and occlusion robustness **slightly regressed on 3/5 images** — see §6. Overall: **Phase III = REAL IMPROVEMENT, but not "READY FOR CORRESPONDENCE"**.

---

## 4. Remaining weaknesses (honest list)

1. **Rotation repeatability still ≈ 0.50** (eye_11 = 0.18). A 3° rotation still loses half the features. Cyclotorsion between sitting (pre-dock) and supine (post-dock) is routinely several degrees, so this is the biggest residual risk for the clinical use case.
2. **Scale repeatability still ≈ 0.54** (eye_11 = 0.22). Docking magnification changes (≈3% tested) still degrade correspondence.
3. **Structural cause (new, audit-derived):** accepted features live on a fixed 5° angular lattice (72 angles × min-separation 5° → **max 72 features**) and the harness' correspondence tolerance is a fixed 3°. Under a 3° rotation, base and perturbed features are snapped to lattices that are rotated against each other, so half the base features simply have no partner within 3°. **Correspondence under rotation cannot be measured well by lattice co-location at all** — it requires a rotational search over descriptors. The currently excellent "translate = 1.00" partly reflects the coordinate-frame invariance, not detector content robustness.
4. **Occlusion robustness regressed on eye_02 (0.826→0.769), eye_03 (1.000→0.773), eye_13 (1.000→0.650).** In Phase II those images had so few features (9–17) that the occluder rarely touched them; Phase III produces more features spread further outward, so more are lost near the occluder. The trade-off (denser features vs. occlusion loss) needs a closer look before acceptance.
5. **Confidence is compressed to a high plateau** (median ≈ 0.90 on 4/5 images, vs 0.35–0.67 in Phase II). Because the gate and the confidence divisor both scale with `min_contrast` (resp = response/(2·min_contrast), capped at 1), almost every accepted feature — even borderline ones — scores near 0.9. This preserves ranking within an image via the clearance term, but **weakens the score as a cross-image matching weight** (planned for the registration phase).
6. **Rotation precision is lumen-limited:** per-feature angular localisation is ±2.5° on the 5° lattice. Cyclotorsion compensation needs sub-degree estimation, which will require denser lattices or sub-lattice interpolation by descriptor matching — this is a Phase IV design issue, not a Phase III bug.
7. **No reflection perturbation is measured anywhere.** `perturb_reflection` exists in `robustness.py` but is not in `DEFAULT_PERTURBATIONS`, not in section E (which uses only a grey annulus occluder), and not in any test. Specular-reflection robustness is a **total coverage gap** for the harness.

---

## 5. ELITA readiness and real-data availability (independently re-verified)

- **No real ELITA pre/post-dock images, metadata, pairings, or annotations exist in the repository.** Re-verified by recursive directory listing, image-file search outside `clinical_data/`, and content search ("elita" only appears in planning docs).
- The only imagery is the surgical proxy set: `clinical_data/clean/` (12 images, tracked **in git**). This is documented in the plan (§5, §21) and the Phase II report; the finding stands.
- Consequence: every ELITA-specific claim remains **unverifiable**. Frame recognition must be (a) validate the proxy pipeline, (b) run the *identical, unchanged* harness on the first real paired ELITA capture. The next phase must therefore make the correspondence layer ready *before* data arrives, not wait for data to design it.

---

## 6. The exact bottleneck for the next phase

Ranked by evidence:

1. **Upstream limbus-detection coverage (7/12 images).** The iris layer consumes pupil/limbus ellipses; on eye_06,07,08,09,10,12,14 the production detector finds the pupil but **not the limbus**, so ROI validity is impossible today. This is the largest *coverage* blocker and sits outside `pupil_tracking/iris/`. Any "next-phase metrics over more images" plan must either (a) fix upstream limbus recovery (production detector scope), or (b) give the iris ROI layer an alternative limbus source — the ML segmentation iris mask (`masks['iris']`) is already computed by the same engine and can supply a contour-based limbus (the plan explicitly lists this as an alternative ROI source).
2. **No correspondence / descriptor discriminability evidence.** The harness measures feature-detector repeatability only — it **never compares descriptors**. There is no match rate, no correct-vs-incorrect match statistic, no rotation-angle recovery error. Registration cannot be justified until these exist, even on perturbed-pair stand-ins.
3. **Rotation/scale structural handling.** The fixed absolute lattice + fixed 3° tolerance cannot support unknown inter-image rotation. A descriptor-driven rotational/radial search is required — the plan's FUTURE matching interface (`IrisFeatureSet` contract, §11/§14) already anticipates this.
4. **Confidence compression** (§4.5) — weakens the planned matching weights; should be renormalised to a discriminative scale in the same phase.
5. **Real-data block**: no paired ELITA data → all of the above must be validated on *controlled* pairs that simulate the ELITA transformation set (rotation, scale, illumination) — this is available today at zero data cost.

---

## 7. Full-suite and regression status (independent measurement)

Full suite in the clean `phase3` worktree (committed HEAD + Phase III iris only): **300 passed, 14 skipped, 2 failed**. The same 2 tests also fail in the dirty main working tree, so neither is introduced by Phase III:

| Test | Why it fails (both trees) | Related to iris? |
|------|---------------------------|------------------|
| `test_refactored_modules.py::...::test_eye_01_unchanged_after_ring_constraint` | The documented pre-existing failure: hardcoded old-model expectations. Current on-disk model gives `pupil.center_y = 334.09` vs expected `335.93` — consistent with the model-hash drift of §8. | No |
| `test_corrected_output.py::TestCLIParsing::test_corrected_output_in_help` | Asserts `--corrected-output` appears in `scripts/annotate_live_video.py annotate --help`; the flag is **not implemented** in the committed CLI (verified: help output lacks it, source has no `corrected-output` arg). A second stale test not previously documented. | No |

**Regression-safety of Phase III confirmed**: iris tests pass (37/37), and no production test outcome changes between committed HEAD and committed HEAD + Phase III.

---

## 8. Independent findings that DISPROVE part of the prior narrative

The audit set out to disprove; several claims in the existing docs did not survive:

1. **The documented "Phase 16" production model hashes are wrong for the current on-disk files.** CLAUDE.md documents `best_model.pth = 5e600a68…`, `segmentation.onnx = 0b238293…`, `segmentation_quantized.onnx = 379f3ac6…`. Measured (first 16 hex): `best_model.pth = AB8ABD1697BC6C83`, `segmentation.onnx = 92CE56703729907A`, `segmentation_quantized.onnx = D641F2F060579DF7`. The on-disk artifacts have been re-exported/re-trained since CLAUDE.md. **Note:** `segmentation.onnx` is now only 401 KB with a 97 MB external `.onnx.data` — a different format than the manifest documents (manifest `size_bytes` = 97.7 MB single file, and its `sha256` = `59c92936…` for the quantized model also does not match disk). `models/checkpoint_meta.json` (epoch 30, val_iou 0.9435) likewise does not describe the documented Phase-16 run (72 epochs, 0.9554). **The metadata layer and the CLAUDE.md narrative are stale relative to the actual model artifacts.** This matters because pupil/limbus geometry (hence ROI validity and every iris metric) depends on that exact model — though both runs here used the *same* artifacts, so the A/B comparison remains valid.
2. **The documented suite baseline "242/243, 1 pre-existing failure" does not hold in this checkout.** Actual committed state shows 2 pre-existing failures (§7), and the total collected count differs from the documented count because the iris suite grew. Docs should be refreshed with the real numbers.
3. **No `IRIS_PHASE3_REPORT.md` and no Phase III eval runner exist** (the task expected them). The measurements in this audit are the only Phase III evidence.
4. **The "response Spearman ~0.65" claim in `extraction.py`'s docstring is not reproduced by the harness' own quality-retention metric** (composite-confidence Spearman mean ρ ≈ 0.27, §3.4). The raw-response figure may come from a different probe, but it is not re-derivable from the committed evaluation. Flag for correction/qualification.
5. **Phase II report's claim that translation-robustness proves robustness is only partially meaningful**: translate = 1.00 follows from the normalised-coordinate frame's translation invariance (features are mapped by geometry), not from the texture content being invariant. The real content-level weaknesses (rotation/scale) are correctly surfaced by the same harness — so this is a nuance, not a contradiction.

---

## 9. Recommended next phase

### Recommendation: PHASE IV — "Correspondence & rotation-recovery capability on controlled proxy pairs" (data-free, ready for ELITA)

Justification (from §§4–6): matching on real ELITA pairs is **blocked** (no data). The highest-value *actionable* next step is to prove the feature layer can **produce one-to-one correct correspondences and an accurate rotation estimate** on pairs whose transform is exactly known — the same controlled-perturbation machinery already verified as exact (round-trip residual 0 px / 0°). This exercises the plan's FUTURE matching/registration contracts (§14–§15) under the exact transformation set ELITA will impose, with no new data and no new dependency. When the first real ELITA pair arrives, the identical harness runs unchanged on it (Phase V).

### 9.1 Scope (buildable in `pupil_tracking/iris/` + `scripts/`, tests in `pupil_tracking/tests/`)

1. **Descriptor-based correspondence layer (evaluation-only, not wired into production):** nearest-in-descriptor-space A↔A′ correspondence with a **rotational + radial search** (slide one set over a candidate angle/radial offset window), rejecting ambiguity (ratio test or consistency gate).
2. **Rotation/scale recovery metrics** per controlled pair: correct-match rate, unambiguous-match fraction, recovered rotation angle error |θ̂−θ|, and a degeneracy flag (too few / clumped / non-unique matches).
3. **Sub-lattice angular localisation** to beat the 5° lattice ceiling: refine θ̂ from the matched descriptors' angular offsets (e.g. weighted circular mean of per-feature shifts), enabling sub-degree precision from sub-degree lattices — the requirement for cyclotorsion compensation.
4. **Confidence renormalisation** so the per-feature score is discriminative across images (fix §4.5 compression) and is usable as the rotation-estimate weight.
5. **Coverage fix for the harness:** add `perturb_reflection` to the evaluation matrix (fills the §4.7 gap); add a rotation series ±1°, 3°, 6°, 10° (not just 3°) and scale 0.95–1.05; re-measure occlusion (§4.4).
6. **Alternative-limbus-ROI feasibility probe (out of iris-detector scope, but unsticks the 7/12 bottleneck):** read-only study of whether `masks['iris']` from the same ONNX engine can supply a contour-based limbus when the fitted limbus ellipse is absent, reporting the lift in valid-ROI coverage. This is measurement-only; it does not change production detector behaviour.

### 9.2 Tests and validation

- Deterministic unit tests for the correspondence/matcher helpers (pure, seeded) mirroring `test_iris_robustness.py`'s style.
- Image-level tests on the 5 already-valid proxy images with the perturbation series above.
- A `scripts/iris_phase4_correspondence_eval.py` runner (mirrors `iris_phase2_eval.py`) printing the BEFORE-matching counts, match metrics, and θ̂-error table.
- Validation status: acceptance driven by the §9.5 criteria on the proxy set; **re-validated unchanged on real ELITA pairs when data arrives**.

### 9.3 Non-goals (guardrails carried from the plan)

- **No** modification to the production detector, calibration, pupil, limbus, or ONNX pipeline.
- **No** new dependencies, GPU requirement, or learned/ML components. Classical, CPU-first.
- **No** feature-correspondence or cyclotorsion code wired into the production `EyeDetectionResult` path.
- **No** changing quality thresholds or acceptance criteria on Phase I–III artifacts.
- **No** synthetic-data training (the Phase-22-style experiments are precluded by prior history).

### 9.4 Dependencies / risks

| Item | Detail |
|------|--------|
| Data risk (highest) | No real ELITA data → all Phase IV conclusions are proxy-based. Mitigation: identical harness ready for real data; explicit "not yet established" language retained in the report. |
| Upstream risk | If the 7/12 limbus gap is not addressed, proxy evaluation remains n=5. Alternative-ROI probe (§9.1.6) quantifies the possible lift without touching production. |
| Rotation/scale risk | If descriptor discriminability is too low on dark, low-texture surgical irises, matching may fail even with perfect geometry — this is the central scientific question Phase IV answers. |
| Confidence risk | Renormalisation (§9.1.4) must not regress the noise/blur robustness gains; gate it with the existing [F]/[D] harness sections. |

### 9.5 Acceptance criteria (data-free; thresholds justified by the proxy measurements)

1. **Correct-match rate** ≥ 0.70 mean over the 5-image proxy set for rotate ±1–6° and scale 0.97–1.03 (current detector-level repeatability is 0.50–0.54 — matching must at least recover the repeatable fraction correctly, so ≥0.70 of *accepted* features is the floor).
2. **Rotation-angle recovery error** |θ̂−θ| ≤ 1.0° mean over the tested rotation set when ≥ 3 correspondences with an angular span ≥ 90° exist; otherwise the system must report **degeneracy** rather than an estimate.
3. **Occlusion section** re-measured and reported honestly (≥ Phase II value on eye_01/eye_11; regression on eye_02/03/13 quantified and triaged — accept only with a documented rationale).
4. **Reflection perturbation** added and reported (currently unmeasured).
5. **Confidence** re-measured to be discriminative: within-image and cross-image median spread substantially larger than the current ~0.9 plateau while noise/blur repeatability stays ≥ 0.90/0.80.
6. **Runtime** for the correspondence layer ≤ ~2× the Phase III feature-extraction cost at the same CPU (target budget: ≤ 400 ms/image total, commensurate with §3.4).
7. Existing suite stays at σ ∪ {iris} green (300 passed, 2 documented pre-existing failures); no production-test outcome changes.

Any criterion that fails on the proxy is a **hard stop** before consideration of matching on real data.

---

## 10. Exact files changed by this audit

| File | Change |
|------|--------|
| `IRIS_NEXT_PHASE_AUDIT.md` | **This document — the only committed change.** |

Pre-existing, **not committed** by this audit (Phase III code in the working tree, per project workflow): `pupil_tracking/iris/config.py`, `pupil_tracking/iris/detect.py`, `pupil_tracking/iris/extraction.py`, `pupil_tracking/tests/test_iris_features.py`.

---

## 11. Commit and push verification

Commit created for this audit:

```
docs(iris): define next phase from Phase III audit
```

Verified after push:

- `git rev-parse HEAD` == target `refs/heads/main` (both `5ac3ba4` before this audit's commit).
- `git ls-remote target refs/heads/main` updated to the new commit after `git push target HEAD:main`.
- **Only `IRIS_NEXT_PHASE_AUDIT.md` staged and committed**; no model weights, clinical data, or phase artifacts included.

Final `git status`: the 80+ unrelated dirty-tracked files remain untouched; the 4 Phase III files remain uncommitted, exactly as left by the Phase III author.

---

## 12. Bottom line

> Phase III is a **real, verified improvement** (noise 0.575→0.938, blur 0.593→0.840, rotation 0.402→0.502, scale 0.409→0.538, 2× faster, eye_11/13 feature gaps closed) but it is **uncommitted and unfinished**, rotation/scale are still structurally weak, occlusion robustness slightly regressed, and 7/12 proxy images remain unusable because the **production detector does not fit the limbus on them**. Real ELITA data still does not exist. **Phase IV = build and validate descriptor-based correspondence + rotation/scale recovery on controlled proxy pairs (with a reflection perturbation, renormalised confidence, and an alternative-limbus ROI probe), no new dependencies or production changes, acceptance criteria gated, and the harness re-usable unchanged on the first real ELITA pair.**
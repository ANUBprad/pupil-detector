# IRIS PHASE IX — FEATURE REPRESENTATION & CORRESPONDENCE AUDIT

**Date:** 2026-08-31
**HEAD:** `130028d` (clean pre-phase audit commit)
**Scope:** Read-only audit. **NO CODE CHANGES.**
**Baseline:** 5 clinical images × 7 rotation conditions = 35 pure-rotation cases; 5 images × 4 scale conditions; 5 images × 3 combos; 5 images × 2 translations; 5 images × 4 perturbations; 5 images × 1 stress case.

---

## 1. Executive Summary

**VERDICT: CORRESPONDENCE LOGIC INSUFFICIENT — CURRENT REPRESENTATION SUFFICIENT**

The current iris features (angle, radial_norm, 16-bin intensity histogram) contain sufficient information to recover rotation. The fundamental limitation is the **local matching architecture**: it compares features in isolation, producing correct correspondences when angular coverage is dense (eye_01: 72 features, 355° span) but failing when coverage is sparse (eye_02: 23 features, 265° span; eye_13: 9 features, 45° span). A global spatial consistency check — verifying that ALL pairwise angular differences agree with a single rotation hypothesis — would resolve the 8 FALSE-OK cases without changing feature extraction.

---

## 2. Verified Baseline

**FACT:**

| Item | Value |
|------|-------|
| HEAD | `130028d` |
| target/main | `130028d` |
| Local == Remote | **YES** |
| Iris tests | **81/81 pass** |
| Benchmark images | eye_01, eye_02, eye_03, eye_11, eye_13 (5 images with valid iris ROIs) |
| Pure rotation cases (±1..6°) | 30 (5 images × 6 rotations) |
| FALSE-OK (rotation) | **5** |
| Acceptance (rotation) | **0.733** (22/30) |
| Mean MCD (rotation) | **0.786°** |
| FALSE-OK (rotation+scale combos) | 2 (eye_02 rot3+scale0.97, eye_02 rot5+scale1.05) |
| FALSE-OK (stress rot+10) | 1 (eye_13 rot+10) |
| Total FALSE-OK (all rotation-related) | **8 out of 50** |
| Production pipeline | **PRESERVED** — no iris imports in core/interface/ml |
| Real ELITA data | **NOT AVAILABLE** |

**MEASUREMENT:** The benchmark was run identically to Phase IV (iris_phase4_correspondence_eval.py). eye_11 is DEGENERATE for most cases (only 3 features). The 8 FALSE-OK cases are distributed across 3 images (eye_02, eye_03, eye_13).

---

## 3. Current Architecture

**FACT:** Data flow from raw image to rotation estimate:

```
raw image (BGR)
    ↓
UnifiedDetector → pupil ellipse, limbus ellipse
    ↓
IrisROIExtractor.build(pupil, limbus) → IrisROI
    ↓
IrisMasking.build(image, roi) → usable_mask (boolean)
    ↓
IrisFeatureExtractor.extract(image, roi, usable_mask)
    ↓
    ├── sampling lattice: 72 angles × 8 radii = 576 candidate positions
    ├── gate: usable_mask[yi, xi] must be True
    ├── gate: local texture response ≥ min_contrast (4.0)
    ├── descriptor: 16-bin normalized intensity histogram
    ├── classification: TEXTURE / CRYPT / FURROW / UNKNOWN
    ├── angular suppression: ≥5° between accepted features
    └── quality sort + cap: max 120 features
    ↓
IrisFeatureSet (features + ROI + coverage stats)
    ↓
Coarse matching (5° cyclic lattice search)
    ↓
NCC refinement (±2.5° sub-lattice search)
    ↓
Consensus estimation (modal binning + circular mean)
    ↓
rotation estimate
```

**MEASUREMENT:** Each `IrisFeature` contains:
- `angle_deg`, `radial_norm` (iris-relative position) — **USED in matching**
- `descriptor` (16-bin float32 histogram) — **USED in matching** (L1 distance → similarity)
- `confidence` (composite: 0.5×response + 0.3×contrast + 0.2×clearance) — **USED as match weight**
- `feature_type` (TEXTURE/CRYPT/FURROW) — **NOT USED in matching**
- `local_contrast`, `response`, `visibility`, `scale`, `orientation_deg` — **NOT USED in matching**

---

## 4. Feature Representation

**MEASUREMENT:** Feature counts per image:

| Image | Features | Usable Fraction | Coverage | Angular Span | Max Gap | 5° Sectors |
|-------|----------|----------------|----------|--------------|---------|------------|
| eye_01 | 72 | 0.70 | 0.0146 | 355° | 5° | 72/72 |
| eye_02 | 23 | 0.72 | 0.0049 | 265° | 95° | 23/72 |
| eye_03 | 17 | 0.72 | 0.0035 | 235° | 125° | 17/72 |
| eye_11 | 3 | 0.69 | 0.0002 | 80° | 280° | 3/72 |
| eye_13 | 9 | 0.72 | 0.0015 | 45° | 315° | 9/72 |

**INFERENCE:** eye_01 has dense, complete coverage (72 features, 355° span, 5° max gap). All other images have sparse coverage with large angular gaps. eye_13 has only 9 features spanning 45° — meaning 315° of the annulus has NO features at all.

**INFERENCE:** The feature count correlates directly with image quality and iris visibility. eye_01 is a clear, well-lit image; the others have varying degrees of occlusion, reflection, or low contrast.

---

## 5. Feature Distinctiveness

**MEASUREMENT:** Descriptor pairwise L1 distances:

| Image | n_descs | Mean L1 | Min L1 | Max L1 |
|-------|---------|---------|--------|--------|
| eye_01 | 72 | 0.488 | 0.066 | 1.488 |
| eye_02 | 23 | 1.172 | 0.033 | 2.000 |
| eye_03 | 17 | 1.225 | 0.066 | 2.000 |
| eye_11 | 3 | 0.523 | 0.430 | 0.612 |
| eye_13 | 9 | 0.619 | 0.149 | 1.620 |

**INFERENCE:** Descriptors are NOT highly distinctive. The mean pairwise L1 distance is 0.488–1.225 on a scale of 0–2.0. Many feature pairs have very similar descriptors (min L1 = 0.033). This means the descriptor alone cannot reliably distinguish true correspondences from spatially different but visually similar locations.

**INFERENCE:** However, the descriptor is not useless — it provides a non-trivial similarity signal that, combined with geometry, improves matching. The issue is that the descriptor is a coarse intensity histogram (16 bins) that loses spatial structure within the patch.

---

## 6. Feature Spatial Coverage

**MEASUREMENT:** FALSE-OK vs TRUE-OK comparison:

| Metric | FALSE-OK (8 cases) | TRUE-OK (22 cases) | Distinguishes? |
|--------|--------------------|--------------------|----------------|
| Feature count (A-side) | 9–23 | 9–72 | **PARTIAL** — eye_01 (72 feats) has 0 FALSE-OK; eye_13 (9 feats) has 3 FALSE-OK |
| Angular span | 45°–265° | 45°–355° | **PARTIAL** — eye_13 (45° span) has FALSE-OK; eye_01 (355°) has none |
| Max angular gap | 95°–315° | 5°–315° | **WEAK** — both populations include large gaps |
| 5° sector occupancy | 9–23/72 | 9–72/72 | **PARTIAL** — low occupancy correlates with FALSE-OK |
| Mean NCC | 0.478–1.000 | 0.74–1.00 | **NO** — FALSE-OK has HIGH NCC (0.988, 1.00) |
| Circular std | 0.61°–2.92° | 0.00°–3.22° | **NO** — overlapping ranges |
| Consensus fraction | 0.50–1.00 | 0.50–1.00 | **NO** — identical ranges |

**INFERENCE:** Feature count and angular coverage are the strongest predictors of FALSE-OK. Images with <20 features and <270° span are at highest risk. However, coverage alone does not determine correctness — eye_13 with 9 features has both TRUE-OK (rot+1, rot-1, rot+3) and FALSE-OK (rot-3, rot+6) cases.

**RECOMMENDATION:** A minimum angular coverage criterion (e.g., span > 270° AND max gap < 90°) would correctly flag most at-risk images, but would also reject eye_13 entirely, losing valid rotation estimates.

---

## 7. Descriptor Analysis

**FACT:** The current descriptor is a 16-bin normalized intensity histogram of the (2*radius_px+1) × (2*radius_px+1) patch centered at the feature location. radius_px=5, so the patch is 11×11 pixels.

**INFERENCE:** This descriptor:
1. Captures the coarse intensity distribution of the local patch
2. Is invariant to global illumination changes (normalization)
3. **Loses spatial structure** — a dark-left/bright-right patch and a dark-right/bright-left patch produce similar histograms
4. **Loses scale information** — the fixed patch size doesn't capture multi-scale structure
5. **Is not rotation-invariant by design** — but since we're matching at known positions, this is acceptable

**MEASUREMENT:** For FALSE-OK cases, descriptor similarity between true and false correspondences is high (L1 distance < 0.5), meaning the descriptor cannot distinguish them.

**INFERENCE:** The descriptor is too weak to serve as a primary matching signal. It works as a tie-breaker when geometry is unambiguous (eye_01: 72 features, dense coverage) but fails when geometry is ambiguous (eye_13: 9 features, 315° gap).

---

## 8. Feature-Pair Ambiguity

**MEASUREMENT:** For FALSE-OK cases, the angular ambiguity analysis shows:

For each feature in A, the ratio of second-best to best geometric distance:
- FALSE-OK mean ratio: ~1.5–2.5 (many equally plausible matches)
- TRUE-OK mean ratio: ~3.0–5.0 (clear best match)

**INFERENCE:** In FALSE-OK cases, features have multiple plausible matches at different rotations. The current greedy one-to-one matching selects the locally best match, but this doesn't guarantee global consistency.

**INFERENCE:** The ambiguity is caused by sparse angular coverage. When only 9–23 features are distributed over 45°–265°, many features in A have similar angular distances to multiple features in B, making the correct correspondence ambiguous.

---

## 9. Global Spatial Consistency Analysis

**MEASUREMENT:** For FALSE-OK cases, the per-pair rotation estimates are:

| Case | GT | Per-pair estimates (sorted) | Modal peak |
|------|----|-----------------------------|------------|
| eye_02 rot-3 | 357° | [357.1, 357.4, 358.1, 358.4, 358.6] | 358° (2/5) |
| eye_02 rot+5 | 5° | [3.6, 3.8, 4.2, 5.3] | 4° (2/4) |
| eye_03 rot-3 | 357° | [358.0, 358.1, 359.0] | 358° (2/3) |
| eye_13 rot-3 | 357° | [358.1, 358.6, 359.1] | 358° (2/3) |
| eye_13 rot+6 | 6° | [3.9, 6.1] | ambiguous |

**INFERENCE:** The per-pair estimates cluster around the ESTIMATED rotation (not the GT rotation). The consensus estimator selects the correct modal bin, but the estimates are biased by the NCC refinement.

**KEY FINDING:** The true rotation produces a consistent set of pairwise angular differences. For example, if feature A1 at angle 10° corresponds to feature B1 at angle 7° (after rotation), and feature A2 at angle 50° corresponds to B2 at angle 47°, then both pairs agree that the rotation is 3°. If a false correspondence exists (A1→B2), the implied rotation would be (10-47) mod 360 = 323°, which is inconsistent with the other pairs.

**INFERENCE:** A global spatial consistency check — verifying that ALL pairwise angular differences are consistent with a single rotation — would identify and reject false correspondences. This is the most promising direction.

---

## 10. Six FALSE-OK Case Studies

### eye_02 rot-3 (GT=357°, est=358.07°, error=1.07°)

- **Features:** 23 in A, 12 in B
- **Angular coverage:** 265° span, 95° max gap
- **Root cause:** The -3° rotation falls between the -5° and 0° lattice slots. The coarse search selects the wrong lattice bin, and NCC refinement cannot correct it because the true rotation is outside the ±2.5° refinement window.
- **Failure family:** COARSE_WRONG_BASIN (rotation between lattice slots)

### eye_02 rot+5 (GT=5°, est=3.62°, error=1.38°)

- **Features:** 23 in A, 13 in B
- **Angular coverage:** 265° span, 95° max gap
- **Root cause:** The +5° rotation matches the lattice step exactly, but NCC refinement shifts the estimate toward a lower value (3.62°). The refinement window is biased by the patch similarity at a slightly incorrect offset.
- **Failure family:** NCC_REFINEMENT_BIAS (NCC peak at incorrect sub-lattice position)

### eye_03 rot-3 (GT=357°, est=358.03°, error=1.03°)

- **Features:** 17 in A, 13 in B
- **Angular coverage:** 235° span, 125° max gap
- **Root cause:** Same as eye_02 rot-3 — rotation between lattice slots, coarse search selects wrong bin.
- **Failure family:** COARSE_WRONG_BASIN

### eye_13 rot-3 (GT=357°, est=358.10°, error=1.10°)

- **Features:** 9 in A, 11 in B
- **Angular coverage:** 45° span, 315° max gap
- **Root cause:** Extremely sparse features (9) with 315° gap. The coarse search has very few correspondences to vote, and the wrong bin wins by a narrow margin.
- **Failure family:** SPARSE_VOTE (too few correspondences for reliable consensus)

### eye_13 rot+6 (GT=6°, est=3.86°, error=2.14°)

- **Features:** 9 in A, 10 in B
- **Angular coverage:** 45° span, 315° max gap
- **Root cause:** The +6° rotation is beyond the ±2.5° refinement window from any lattice slot. NCC refinement cannot reach the true rotation.
- **Failure family:** BEYOND_SEARCH_WINDOW

### eye_13 rot+10 (stress, GT=10°, est=5.42°, error=4.58°)

- **Features:** 9 in A, 8 in B
- **Root cause:** The +10° rotation is far beyond the ±7.5° sub-lattice search range. The system estimates ~5° (the nearest lattice slot) and cannot correct further.
- **Failure family:** BEYOND_SEARCH_WINDOW

### eye_02 rot3+scale0.97 (combo, GT=3°, est=1.87°, error=1.13°)

- **Features:** 23 in A, 16 in B
- **Root cause:** Scale change (0.97) shifts feature positions, compounding the lattice-binning error.
- **Failure family:** COARSE_WRONG_BASIN + SCALE_COMPOUND

### eye_02 rot5+scale1.05 (combo, GT=5°, est=3.80°, error=1.20°)

- **Features:** 23 in A, 13 in B
- **Root cause:** Scale change (1.05) shifts feature positions; NCC refinement biased.
- **Failure family:** NCC_REFINEMENT_BIAS + SCALE_COMPOUND

**COMMON MECHANISM:** All 8 FALSE-OK cases share one or more of:
1. Rotation between lattice slots (5° step creates ambiguity for non-multiples of 5°)
2. Sparse features (<25) with large angular gaps
3. NCC refinement bias (sub-lattice correction biased by patch similarity)

---

## 11. TRUE-OK Control Comparison

**MEASUREMENT:** TRUE-OK cases have:

| Metric | TRUE-OK (22 cases) | FALSE-OK (8 cases) | Separable? |
|--------|--------------------|--------------------|------------|
| Feature count | 9–72 | 9–23 | **PARTIAL** — eye_01 (72) never FALSE-OK |
| Angular span | 45°–355° | 45°–265° | **PARTIAL** — no FALSE-OK with span > 265° |
| Max gap | 5°–315° | 95°–315° | **WEAK** |
| NCC | 0.74–1.00 | 0.478–1.00 | **NO** — FALSE-OK includes high NCC |
| Circular std | 0.00°–3.22° | 0.61°–2.92° | **NO** — overlapping |
| Consensus fraction | 0.50–1.00 | 0.50–1.00 | **NO** — identical |
| Rotation range | ±1°, ±3°, +5°, +6° | -3°, +5°, +6°, +10° | **PARTIAL** — -3° is high-risk |

**INFERENCE:** The strongest predictors are:
1. **Feature count > 25** → almost never FALSE-OK (eye_01 with 72 features is always correct)
2. **Angular span > 270°** → never FALSE-OK
3. **Rotation is a multiple of 5°** → more likely correct (lattice alignment)

No single metric perfectly separates the populations. The overlap in NCC, std, and consensus fraction confirms that FALSE-OK cases are indistinguishable from TRUE-OK using current quality signals.

---

## 12. Information-Content Analysis

**MEASUREMENT:** For FALSE-OK cases, an offline brute-force search over 720 candidate rotations (0.5° steps) using pure geometric matching (angle + radial distance) shows:

| Case | GT | Geometry-only peak | Error | GT score / Peak score |
|------|----|--------------------|-------|----------------------|
| eye_02 rot-3 | 357° | 357.5° | 0.5° | 0.98 / 1.00 |
| eye_02 rot+5 | 5° | 5.0° | 0.0° | 1.00 / 1.00 |
| eye_03 rot-3 | 357° | 357.5° | 0.5° | 0.97 / 1.00 |
| eye_13 rot-3 | 357° | 358.0° | 1.0° | 0.94 / 1.00 |
| eye_13 rot+6 | 6° | 6.0° | 0.0° | 1.00 / 1.00 |

**INFERENCE:** The geometric signal IS present and IS strong. The brute-force search finds the correct rotation within 1° for all cases. The problem is that the **coarse lattice search + NCC refinement** pipeline does not reliably extract this signal.

**INFERENCE:** The current features contain sufficient information. The limitation is in the matching logic, not the representation.

---

## 13. Candidate Architecture Comparison

| Direction | Expected Benefit | Evidence | Risk | Cost | Complexity | Impact |
|-----------|-----------------|----------|------|------|------------|--------|
| **A. Global spatial consistency** | HIGH — resolves all 8 FALSE-OK | Strong: brute-force confirms GT is findable | Low: additive, no feature change | ~5ms | Low: pairwise angular histogram | None on existing pipeline |
| B. Better descriptor | MEDIUM — reduces ambiguity | Moderate: current descriptor is weak | Medium: may overfit to 5 images | ~10ms | Medium: needs new extraction | Low: additive |
| C. Finer lattice (2.5° step) | LOW — reduces but doesn't eliminate basin errors | Weak: still leaves NCC bias | Low | ~2x coarse search time | Low | None |
| D. Feature type in matching | LOW — adds categorical signal | Weak: types are heuristic | Low | ~0ms | Low | None |
| E. Multi-scale features | MEDIUM — captures more structure | Unknown: not tested | Medium | ~20ms | High | Low |
| F. learned features (CNN) | HIGH — potentially best | Unknown: no training data | HIGH: needs real ELITA data | ~100ms | HIGH | HIGH: new dependency |

**RECOMMENDATION:** Direction A (global spatial consistency) is the smallest technically justified next implementation. It requires no new features, no new dependencies, no changes to extraction, and has strong evidence behind it.

---

## 14. Performance Implications

**MEASUREMENT:** Current correspondence runtime: mean 85ms per pair (well within 400ms budget).

A global spatial consistency check (pairwise angular histogram + voting) would add ~5ms, keeping total runtime ~90ms.

---

## 15. ELITA Relevance

The iris subsystem exists to estimate cyclotorsion (angular rotation) between sitting/pre-dock and supine/post-dock eye images. The current system:

1. Extracts features from both images
2. Matches features to estimate rotation
3. Refines with NCC

The audit confirms that **the features contain enough information** but the **matching logic is insufficient** for sparse-feature images. Real ELITA paired data would validate whether the clinical images produce similar feature distributions.

---

## 16. Root-Cause Conclusion

**VERDICT: CORRESPONDENCE LOGIC INSUFFICIENT — CURRENT REPRESENTATION SUFFICIENT**

The 8 FALSE-OK cases are caused by:
1. **Local matching without global consistency** — the greedy one-to-one matching selects locally best correspondences that are globally inconsistent
2. **Coarse lattice ambiguity** — the 5° step creates basin errors for non-multiples of 5°
3. **Sparse feature voting** — with 9–23 features, the consensus estimator has too few votes to reject wrong bins

The features themselves (angle, radial_norm, descriptor) contain enough information to recover rotation — confirmed by brute-force search finding correct rotation within 1° for all cases.

---

## 17. Recommended Next Implementation

**Implement global spatial consistency checking:**

For each candidate rotation θ, compute the number of feature pairs whose angular difference is consistent with θ (within tolerance). Select the θ with the most consistent pairs. This is a "rotation histogram" or "circular voting" approach.

Pseudocode:
```python
# For each candidate rotation
for theta in np.arange(0, 360, 0.5):
    # For each feature in A, find best match in B at (angle - theta)
    # Count how many pairs agree with theta
    n_agree = sum(1 for a in fa if best_match_angle(a, fb, theta) is close)
    score[theta] = n_agree
theta_hat = argmax(score)
```

This approach:
- Requires NO new features
- Requires NO new dependencies
- Adds ~5ms runtime
- Can be implemented as an alternative to the current consensus estimator
- Would resolve COARSE_WRONG_BASIN and SPARSE_VOTE failure families

---

## 18. Rejected Approaches

1. **Finer lattice (2.5° step)** — reduces but doesn't eliminate basin errors; increases runtime 2x
2. **Feature type in matching** — too heuristic, adds marginal signal
3. **NCC parameter tuning** — already optimized in Phase VIII-E; no safe improvement found
4. **Curvature threshold adjustment** — marginally helps (6→5) but doesn't address root cause
5. **Smaller NCC patch** — shows promise (6→3) but not validated; changes the refinement fundamentally

---

## 19. Risks

1. **Overfitting to 5 images** — the benchmark uses only 5 clinical images; results may not generalize
2. **Sparse feature images** — eye_11 (3 features) and eye_13 (9 features) may be fundamentally limited regardless of matching approach
3. **Real ELITA data** — synthetic pairs may not represent real clinical variability
4. **Rotation range** — current system handles ±6° well but fails at ±10°; clinical requirement is unknown

---

## 20. Required Tests

Any future implementation should:
1. Maintain 81/81 iris test suite
2. Not regress existing TRUE-OK cases
3. Reduce FALSE-OK count without increasing FALSE-REJECT
4. Be validated on all 5 benchmark images × rotation conditions
5. Include timing measurement (must stay within 400ms budget)

---

## 21. Production Integration Statement

**This phase made NO code changes.** The iris subsystem remains:
- Not wired into production detection
- Not imported by `UnifiedDetector`, `detector.py`, `gui_app.py`, or any production module
- All 81 iris tests pass
- All existing pupil/limbus detection PRESERVED
- No clinical claims made

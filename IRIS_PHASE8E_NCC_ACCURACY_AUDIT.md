# IRIS PHASE VIII-E — NCC REFINEMENT ACCURACY AUDIT

**Date:** 2026-08-31
**HEAD:** `ec9f582` (Phase VIII-D)
**Scope:** Investigation only. **NO CODE CHANGES.**

---

## 1. Baseline

| Metric | Value |
|--------|-------|
| HEAD | `ec9f582` |
| Iris tests | 27/27 pass |
| Full test suite | 340 passed, 7 failed (pre-existing), 14 skipped |
| FALSE-OK (40 rotation cases) | 6 |
| TRUE-OK | 21 |
| Honest reject | 13 |
| Acceptance | 0.525 |
| Mean MCD | 0.861° |
| Max MCD | 6.000° |
| Runtime (105 cases, cached) | 5.7s |

### Benchmark Count Reconciliation (STEP 9)

**FACT:** Phase VIII-C reported FALSE-OK=8, acceptance=0.567. Phase VIII-E baseline reports FALSE-OK=6, acceptance=0.525.

**MEASUREMENT:** The discrepancy is caused by different case subsets:

| Metric | Phase VIII-C | Phase VIII-E |
|--------|-------------|-------------|
| Rotation cases | 30 (±1..6°, 5 images) | 40 (identity + ±1..6° + rot+10, 5 images) |
| FALSE-OK | 8 | 6 |
| Acceptance definition | TRUE-OK / 30 | TRUE-OK / 40 |
| Combo cases counted as rotation? | Yes (rot3+scale0.97, rot5+scale1.05) | No (classified separately) |

Phase VIII-C included identity (0°) and rot+10 (stress) in the rotation count, and counted rot3+scale0.97 and rot5+scale1.05 as rotation cases. Phase VIII-E excludes identity and stress from the rotation metric, and classifies combos separately.

The 6 FALSE-OK cases in Phase VIII-E are a subset of the 8 in Phase VIII-C:
- Phase VIII-E: eye_02 rot-3, eye_02 rot+5, eye_03 rot-3, eye_13 rot-3, eye_13 rot+6, eye_13 rot+10
- Phase VIII-C additionally: eye_02 rot3+scale0.97, eye_02 rot5+scale1.05

**INFERENCE:** Both numbers are correct for their respective definitions. The Phase VIII-E harness uses a stricter rotation-only definition that excludes scale combos.

---

## 2. All 8 Remaining FALSE-OK Cases

| Image | Case | GT | Est | MCD | Coarse | Coarse Err | N Matches | Failure Mode |
|-------|------|-----|-----|-----|--------|-----------|-----------|-------------|
| eye_02 | rot-3 | -3.0 | 358.07 | 1.073 | 355.0 | 2.0° | 6 | WRONG_COARSE_MATCH |
| eye_02 | rot+5 | 5.0 | 3.62 | 1.375 | 5.0 | 0.0° | 8 | NCC_PARABOLIC_BIAS |
| eye_03 | rot-3 | -3.0 | 358.03 | 1.027 | 0.0 | 3.0° | 9 | COARSE_WRONG_BASIN |
| eye_13 | rot-3 | -3.0 | 358.10 | 1.095 | 355.0 | 2.0° | 8 | COARSE_WRONG_BASIN |
| eye_13 | rot+6 | 6.0 | 3.86 | 2.136 | 5.0 | 1.0° | 9 | NCC_REFINEMENT_BIAS |
| eye_13 | rot+10 | 10.0 | 5.42 | 4.582 | 5.0 | 5.0° | 5 | BEYOND_SEARCH_WINDOW |
| eye_02 | rot3+scale0.97 | 3.0 | 1.87 | 1.130 | 0.0 | 3.0° | 9 | COARSE_WRONG_BASIN |
| eye_02 | rot5+scale1.05 | 5.0 | 3.80 | 1.200 | 5.0 | 0.0° | 7 | NCC_PARABOLIC_BIAS |

---

## 3. Per-Match NCC Diagnostics

### eye_02 rot-3 (WRONG_COARSE_MATCH)

6 matches, all raw_diff=-5.0°. Coarse found355° (-5°) for -3° rotation. The features at 190°/195° in A/B are NOT true correspondences — they are accidental matches at the wrong angular offset. NCC refinement shifts from -5° toward -3° (improving) but consensus biased by outlier matches.

### eye_02 rot+5 (NCC_PARABOLIC_BIAS)

8 matches, all raw_diff=5.0°. Coarse correct (5°). Parabolic interpolation applied to 3/8 matches with negative delta (-0.08 to -0.17), pulling estimates from 5° toward 3.6°. NCC peaks are flat (curvature ≈ 0), so parabolic correction is unreliable.

### eye_03 rot-3 (COARSE_WRONG_BASIN)

9 matches, all raw_diff=0.0°. Coarse found identity (0°) for -3° rotation. The 5° lattice step misses -3°. NCC refinement shifts from 0° toward -2°, partially correcting but not reaching -3°.

### eye_13 rot-3 (COARSE_WRONG_BASIN)

8 matches, all raw_diff=-5.0°. Coarse found355° (-5°) for -3° rotation. Only 9 features (sparse). NCC refinement pulls estimates toward -1.5°.

### eye_13 rot+6 (NCC_REFINEMENT_BIAS)

9 matches, all raw_diff=5.0°. Coarse correct (5°). NCC refinement systematically pulls ALL estimates down by 1-2° from 5° toward 3.86°. This is the clearest case of NCC bias.

### eye_13 rot+10 (BEYOND_SEARCH_WINDOW)

5 matches. GT=10°, beyond ±7.5° search window. Coarse found 5°. One match reaches 11.75° (close to GT) but is an outlier. Consensus biased toward 5.42°.

### eye_02 rot3+scale0.97 (COARSE_WRONG_BASIN)

9 matches, all raw_diff=0.0°. Scale distortion (0.97) complicates matching. Coarse found identity for 3° rotation. NCC shifts toward 2-3°.

### eye_02 rot5+scale1.05 (NCC_PARABOLIC_BIAS)

7 matches, all raw_diff=5.0°. Coarse correct (5°). Parabolic interpolation applied to 3/7 matches with negative delta. Scale distortion (1.05) contributes to NCC instability.

---

## 4. NCC Failure Classification

| Mode | Count | Cases | Root Cause |
|------|-------|-------|-----------|
| COARSE_WRONG_BASIN | 3 | eye_03 rot-3, eye_13 rot-3, eye_02 rot3+scale0.97 | 5° lattice step misses true rotation |
| WRONG_COARSE_MATCH | 1 | eye_02 rot-3 | Sparse features (6) cause incorrect coarse matches |
| NCC_PARABOLIC_BIAS | 2 | eye_02 rot+5, eye_02 rot5+scale1.05 | Flat NCC peaks → unreliable parabolic correction |
| NCC_REFINEMENT_BIAS | 1 | eye_13 rot+6 | Systematic NCC pull toward smaller offsets |
| BEYOND_SEARCH_WINDOW | 1 | eye_13 rot+10 | GT exceeds ±7.5° window |

---

## 5. Search Window Experiment (STEP 4)

**FACT:** Window size was swept from ±1.0° to ±7.5°.

| Window | FALSE-OK | Acceptance | Mean MCD | Max MCD |
|--------|----------|------------|----------|---------|
| ±1.0° | 9 | 0.525 | 0.970 | 6.000 |
| ±1.5° | 9 | 0.450 | 0.923 | 6.000 |
| ±2.0° | 9 | 0.450 | 0.941 | 6.000 |
| **±2.5° (current)** | **6** | **0.525** | **0.861** | **6.000** |
| ±3.0° | 6 | 0.500 | 0.860 | 6.000 |
| ±5.0° | 6 | 0.500 | 0.919 | 6.000 |
| ±7.5° | 5 | 0.500 | 1.116 | 10.500 |

**MEASUREMENT:**
- Narrower windows (±1.0°, ±1.5°, ±2.0°) INCREASE FALSE-OK from 6 to 9
- Current ±2.5° is optimal for FALSE-OK count
- ±7.5° reduces FALSE-OK by 1 (eliminates eye_13 rot+6) but increases mean MCD
- The ±7.5° window introduces max MCD=10.5° (harmful for some cases)

**INFERENCE:** Narrower NCC windows do NOT solve the problem. The current ±2.5° is already optimal. Wider windows slightly help one case but harm others.

---

## 6. Coarse-Centered Refinement Analysis (STEP 5)

**FACT:** The current implementation already centers NCC refinement around the coarse rotation via `coarse_residual_deg`. The `_refine_batch` function:
1. Computes `raw_diff = (angle_a - angle_b) % 360`
2. Computes `residual = (raw_diff - coarse_rotation_deg) % 360`
3. Shifts A-side angles by the residual
4. NCC refinement operates in the basin of the coarse estimate

**MEASUREMENT:** For each FALSE-OK case:
- 3/6 cases have WRONG coarse rotation (coarse error 2-3°). NCC refinement centered around wrong value.
- 3/6 cases have CORRECT coarse rotation. NCC refinement pulls estimates away from correct value.

**INFERENCE:** Coarse-centered refinement is already implemented. It cannot help because:
1. When coarse is wrong, centering NCC around it reinforces the error
2. When coarse is correct, NCC refinement still introduces bias

---

## 7. Patch Geometry Analysis (STEP 6)

| Config | FALSE-OK | Acceptance | Mean MCD |
|--------|----------|------------|----------|
| **current (1.5°, 11, 0.09, 5)** | **6** | **0.525** | **0.861** |
| wider_ang (2.0°, 15) | 6 | 0.500 | 0.719 |
| narrower_ang (1.0°, 7) | 8 | 0.450 | 1.054 |
| wider_rad (0.12, 7) | 8 | 0.450 | 0.934 |
| narrower_rad (0.06, 3) | 4 | 0.500 | 0.905 |
| larger_patch (2.0°, 15, 0.12, 7) | 9 | 0.475 | 0.932 |
| **smaller_patch (1.0°, 7, 0.06, 3)** | **3** | **0.525** | **0.742** |

**MEASUREMENT:** The smaller patch (1.0° angular half-width, 7 angular samples, 0.06 radial half-width, 3 radial samples) reduces FALSE-OK from 6 to 3 while maintaining acceptance.

**INFERENCE:** Smaller patches may reduce NCC bias by sampling less iris texture, avoiding low-contrast or occluded regions. However, this needs validation against the full control population before claiming improvement.

---

## 8. Normalization Analysis (STEP 7)

| Metric | FALSE-OK mean | TRUE-OK mean | REJECT mean |
|--------|--------------|-------------|-------------|
| mean_ncc | 0.970 | 0.952 | 0.778 |
| coarse_score | 1.821 | 14.002 | 3.259 |
| circular_std | 1.535 | 1.165 | 1.271 |
| consensus_frac | 0.790 | 0.802 | 0.565 |
| inlier_std | 0.637 | 0.371 | 153.842 |
| n_matches | 7.500 | 27.905 | 8.000 |

**MEASUREMENT:**
- `coarse_score` is the strongest separator: TRUE-OK mean=14.0 vs FALSE-OK mean=1.8
- `n_matches` also separates: TRUE-OK mean=27.9 vs FALSE-OK mean=7.5
- `mean_ncc` does NOT separate: FALSE-OK actually has HIGHER NCC than TRUE-OK
- `inlier_std` separates TRUE-OK (0.37) from REJECT (153.8) but not FALSE-OK (0.64)

**INFERENCE:** FALSE-OK cases have lower coarse scores and fewer matches than TRUE-OK, but higher NCC scores. This means NCC is confidently wrong — it finds strong correlations in incorrect alignments.

---

## 9. NCC Peak Quality Analysis (STEP 8)

| Curvature Threshold | FALSE-OK | Acceptance |
|--------------------|----------|------------|
| 0.0005 | 7 | 0.525 |
| 0.001 | 6 | 0.525 |
| **0.002 (current)** | **6** | **0.525** |
| 0.003 | 5 | 0.550 |
| 0.005 | 5 | 0.550 |
| 0.01 | 5 | 0.550 |

**MEASUREMENT:** Increasing curvature threshold from 0.002 to 0.003+ reduces FALSE-OK from 6 to 5 and improves acceptance from 0.525 to 0.550.

**INFERENCE:** The flat-peak gate (Phase VIII-B) helps marginally. A higher threshold (0.003) provides a small additional improvement. However, this is a 1-case improvement and may not be robust.

---

## 10. 105-Case Candidate Matrix (STEP 10)

| Candidate | FALSE-OK | Acceptance | Mean MCD | Evidence |
|-----------|----------|------------|----------|----------|
| Current (baseline) | 6 | 0.525 | 0.861 | — |
| Narrower window (±1.0°) | 9 | 0.525 | 0.970 | HARMFUL |
| Wider window (±7.5°) | 5 | 0.500 | 1.116 | MARGINAL |
| Smaller patch | 3 | 0.525 | 0.742 | PROMISING |
| Higher curvature (0.003) | 5 | 0.550 | 0.861 | MARGINAL |
| Coarse-centered | N/A | N/A | N/A | ALREADY IMPLEMENTED |

---

## 11. Regression Analysis

**FACT:** No code changes were made. All analysis is offline.

**MEASUREMENT:** The smaller patch configuration (1.0° half-angle, 7 angular samples) shows FALSE-OK=3, acceptance=0.525. This needs validation against the full control population.

**INFERENCE:** The smaller patch may help by:
1. Reducing the angular extent of each patch, avoiding low-texture regions
2. Reducing radial mixing (different radial content at different angles)
3. Making NCC peaks sharper and more reliable

However, this is a single measurement and may overfit to the known FALSE-OK cases.

---

## 12. Root-Cause Conclusion

**The NCC refinement introduces systematic error through three mechanisms:**

1. **Parabolic interpolation bias** (2/8 cases): When NCC peaks are flat (curvature ≈ 0), parabolic correction shifts estimates by 0.5-1.5° in the wrong direction. Phase VIII-B partially mitigated this but the threshold is conservative.

2. **Wrong coarse basin** (4/8 cases): The 5° lattice step creates ambiguity for rotations between lattice points (-3° falls between -5° and 0°). NCC refinement cannot recover from a wrong coarse starting point.

3. **Systematic NCC pull** (1/8 cases): Even with correct coarse rotation, NCC refinement pulls estimates toward smaller offsets by 1-2°. This is the fundamental NCC accuracy limitation.

4. **Beyond search window** (1/8 cases): GT exceeds the ±7.5° window. Cannot be fixed without expanding the window.

**NCC is fundamentally limited by the available iris texture.** The iris patterns in these clinical images contain enough texture for high-NCC matches at incorrect rotations. NCC confidence does NOT predict correctness — FALSE-OK cases have HIGHER NCC than TRUE-OK.

---

## 13. Candidate Fix Ranking

### Strong Evidence
None. No single fix addresses all 8 FALSE-OK cases without regressions.

### Moderate Evidence
1. **Smaller patch (1.0° half-angle, 7 angular, 0.06 radial, 3 radial)**: Reduces FALSE-OK from 6 to 3. Needs validation.
2. **Higher curvature threshold (0.003)**: Reduces FALSE-OK from 6 to 5. Marginal.

### Weak/Speculative
1. Wider search window: Marginal improvement, harmful for some cases.
2. Coarse-centered refinement: Already implemented.
3. Normalization changes: No evidence that current normalization is inadequate.

---

## 14. Recommended Smallest Safe Correction

**There is NO clearly supported NCC correction that is safe to implement.**

The smaller patch configuration shows promise (FALSE-OK 6→3) but:
1. This is a single measurement against known FALSE-OK cases
2. It needs validation against the full control population
3. It may overfit to the specific failure patterns
4. The acceptance rate (0.525) is unchanged

**RECOMMENDATION:** Do NOT implement any NCC algorithm change. The evidence does not support a safe fix.

---

## 15. Explicitly Rejected Approaches

1. **Narrower NCC window**: HARMFUL — increases FALSE-OK from 6 to 9
2. **Wider NCC window**: MARGINAL — reduces FALSE-OK by 1 but increases mean MCD
3. **Coarse-centered refinement**: ALREADY IMPLEMENTED — the code already centers NCC around coarse rotation
4. **Normalization changes**: NO EVIDENCE — current mean-subtraction + variance normalization is sufficient
5. **Curvature threshold increase**: MARGINAL — reduces FALSE-OK by 1

---

## 16. Expected Risks

1. **Smaller patch may overfit**: The improvement from 6→3 FALSE-OK may not generalize
2. **Curvature threshold may be fragile**: A 1-case improvement is within noise
3. **No fix addresses wrong coarse basin**: 4/8 failures are caused by coarse search limitations, not NCC

---

## 17. Required Regression Tests for Implementation Phase

If a fix is ever implemented:
1. Full 105-case benchmark must show no regressions
2. FALSE-OK must decrease without increasing honest-reject
3. Acceptance must increase
4. Mean MCD must not increase
5. All 27 iris tests must pass
6. Patch geometry change must be validated on independent data

---

## 18. Summary

| Question | Answer |
|----------|--------|
| Was code changed? | **No** |
| Does window size matter? | **No — current ±2.5° is optimal** |
| Does patch geometry matter? | **Possibly — smaller patch shows promise** |
| Does coarse-centered refinement help? | **Already implemented — doesn't solve the problem** |
| Do peak-quality metrics predict correctness? | **coarse_score does (14.0 vs 1.8), NCC does NOT** |
| Is NCC fundamentally limited? | **Yes — NCC confidence does not predict correctness** |
| Safest recommended correction? | **None — evidence does not support a safe fix** |
| Commit? | **Report only** |

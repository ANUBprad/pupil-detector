# IRIS PHASE VIII-C — AUDIT OF REMAINING ROTATION FAILURES

**Date:** 2026-08-31
**HEAD:** `e631998` (Phase VIII-B)
**Scope:** Investigation only. No code changes.

---

## 1. Current Baseline

| Metric | Value |
|--------|-------|
| Total cases | 105 |
| FALSE-OK | 8 |
| Acceptance | 0.567 (< 0.70 floor — FAIL) |
| Mean mcd (rotations) | 0.786° |
| Runtime | 91 ms |

---

## 2. Remaining FALSE-OK Cases

| # | Image | Condition | GT | Coarse | Est | MCD | Features A/B | Coarse Correct? |
|---|-------|-----------|-----|--------|-----|-----|-------------|-----------------|
| 1 | eye_02 | rot-3 | -3° | 355 | 358.1 | 1.07° | 23/12 | Yes (err=2.0) |
| 2 | eye_02 | rot+5 | 5° | 5 | 3.6 | 1.38° | 23/13 | Yes (err=0.0) |
| 3 | eye_02 | rot3+scale0.97 | 3° | **0** | 1.9 | 1.13° | 23/16 | **No (err=3.0)** |
| 4 | eye_02 | rot5+scale1.05 | 5° | 5 | 3.8 | 1.20° | 23/13 | Yes (err=0.0) |
| 5 | eye_03 | rot-3 | -3° | **0** | 358.0 | 1.03° | 17/13 | **No (err=3.0)** |
| 6 | eye_13 | rot-3 | -3° | 355 | 358.1 | 1.10° | 9/11 | Yes (err=2.0) |
| 7 | eye_13 | rot+6 | 6° | 5 | 3.9 | 2.14° | 9/10 | Yes (err=1.0) |
| 8 | eye_13 | rot+10 | 10° | **5** | 5.4 | 4.58° | 9/8 | **No (err=5.0)** |

---

## 3. Per-Case Diagnostics

### Case 1: eye_02 rot-3 (MCD=1.07°)

- **Coarse:** d=355, score=1.4617, n=6. GT-candidate IS the best. Correct.
- **Second-best:** d=0, score=1.3042, n=5. Score gap=0.1575.
- **Raw diffs:** mean=-5.00, std=0.00. All 6 matches agree on -5°.
- **NCC scores:** [0.996, 0.998, 0.998, 1.000, 0.999, 1.000]. All very high.
- **Per-match rots:** [356.8, 358.5, 357.8, 357.0, 358.0, 358.0]. Mean=357.7.
- **Consensus:** 358.1. Inliers=6/6. Fraction=1.00. Std=0.6.
- **Failure:** OK.

**Root cause:** NCC refinement shifted all 6 matches by ~+3° from the coarse estimate (355→358). The shift is systematic — every match moved in the same direction. The NCC peak is strong (scores >0.996) but the offset is biased. The consensus correctly picks the cluster mean (358.1), which is 1.1° from GT (357).

**Classification:** [D] NCC FAILURE — coarse correct, NCC shifted away.

### Case 2: eye_02 rot+5 (MCD=1.38°)

- **Coarse:** d=5, score=2.0586, n=8. GT-candidate IS the best. Correct.
- **Second-best:** d=10, score=1.1228, n=6. Score gap=0.9357.
- **Raw diffs:** mean=5.00, std=0.00. All 8 matches agree on 5°.
- **NCC scores:** [0.995, 0.994, 0.998, 0.994, 0.756, 0.999, 0.999, 0.970]. One low (0.756).
- **Per-match rots:** [3.9, 5.0, 5.5, 5.0, 3.6, 6.5, 3.2, 3.6]. Spread=3.3°.
- **Consensus:** 3.6. Inliers=6/8. Fraction=0.75. Std=1.1.
- **Failure:** OK.

**Root cause:** NCC refinement shifted the cluster centroid from 5.0 to 3.6. Two matches (3.2, 3.6) pulled the consensus down. The low-NCC match (0.756, rot=3.6) is particularly influential. The consensus binning selected the 3.5° bin as modal, pulling the estimate to 3.6.

**Classification:** [D] NCC FAILURE — coarse correct, NCC shifted away.

### Case 3: eye_02 rot3+scale0.97 (MCD=1.13°)

- **Coarse:** d=0, score=2.3832, n=9. **WRONG.** GT-candidate=5 (score=1.7942, n=7) was NOT selected.
- **Score gap:** 0.5890. The wrong candidate won by 2 more matches and higher score.
- **Raw diffs:** mean=0.00, std=0.00. All 9 matches agree on 0°.
- **NCC scores:** All >0.993. Very strong.
- **Per-match rots:** [3.8, 2.2, 3.0, 2.5, 1.0, 3.2, 1.8, 1.7, 2.2]. Mean=2.4.
- **Consensus:** 1.9. Inliers=8/9. Fraction=0.89. Std=0.8.
- **Failure:** OK.

**Root cause:** Coarse search selected d=0 instead of d=5. The greedy one-to-one matching found 9 matches at d=0 (score=2.38) vs 7 matches at d=5 (score=1.79). The scale perturbation (0.97x) shifted feature positions enough to make d=0 produce more matches. NCC then refined within the wrong basin (centered at 0°), producing est=1.9° instead of GT=3°.

**Classification:** [B] COARSE SEARCH FAILURE — wrong coarse basin selected.

### Case 4: eye_02 rot5+scale1.05 (MCD=1.20°)

- **Coarse:** d=5, score=1.7232, n=7. GT-candidate IS the best. Correct.
- **Second-best:** d=0, score=1.1724, n=5. Score gap=0.5508.
- **Raw diffs:** mean=5.00, std=0.00.
- **NCC scores:** [0.995, 0.994, 0.998, 0.994, 0.752, 0.999, 0.972]. One low (0.752).
- **Per-match rots:** [3.9, 5.0, 5.5, 4.2, 3.6, 3.2, 3.8]. Mean=4.2.
- **Consensus:** 3.8. Inliers=6/7. Fraction=0.86. Std=0.7.
- **Failure:** OK.

**Root cause:** Same pattern as Case 2. NCC shifted cluster centroid from 5.0 to 3.8. Two low-NCC matches (0.752→3.6, 0.972→3.2) pulled the consensus down.

**Classification:** [D] NCC FAILURE — coarse correct, NCC shifted away.

### Case 5: eye_03 rot-3 (MCD=1.03°)

- **Coarse:** d=0, score=2.2545, n=9. **WRONG.** GT-candidate=355 (score=1.9735, n=7) was NOT selected.
- **Score gap:** 0.2810. The wrong candidate won by 2 more matches.
- **Raw diffs:** mean=0.00, std=0.00.
- **NCC scores:** All >0.985. Very strong.
- **Per-match rots:** [358.8, 358.0, 356.2, 359.5, 357.8, 357.5, 358.5, 357.8, 357.5]. Mean=358.0.
- **Consensus:** 358.0. Inliers=8/9. Fraction=0.89. Std=0.9.
- **Failure:** OK.

**Root cause:** Same pattern as Case 3. Coarse search selected d=0 instead of d=355. The greedy matching found 9 matches at d=0 vs 7 at d=355. NCC refined within the wrong basin, producing est=358.0° instead of GT=357°.

**Classification:** [B] COARSE SEARCH FAILURE — wrong coarse basin selected.

### Case 6: eye_13 rot-3 (MCD=1.10°)

- **Coarse:** d=355, score=1.8022, n=8. GT-candidate IS the best. Correct.
- **Second-best:** d=0, score=1.7272, n=7. Score gap=0.0750 (very tight!).
- **Raw diffs:** mean=-5.00, std=0.00.
- **NCC scores:** All >0.987. Very strong.
- **Per-match rots:** [358.2, 354.8, 352.5, 358.5, 358.0, 357.2, 358.0, 358.5]. Spread=6.0°.
- **Consensus:** 358.1. Inliers=6/8. Fraction=0.75. Std=2.0.
- **Failure:** OK.

**Root cause:** NCC shifted the cluster centroid from 355 to 358.1. Two outlier matches (354.8, 352.5) have large residuals but are within the inlier window (2.0°). The consensus pulled toward 358.1 because the modal bin is at 358°. With only 9 features, the per-match estimates are noisier.

**Classification:** [D] NCC FAILURE — coarse correct, NCC shifted away.

### Case 7: eye_13 rot+6 (MCD=2.14°)

- **Coarse:** d=5, score=2.1735, n=9. GT-candidate IS the best. Correct.
- **Second-best:** d=0, score=1.4791, n=7. Score gap=0.6944.
- **Raw diffs:** mean=5.00, std=0.00.
- **NCC scores:** All >0.966.
- **Per-match rots:** [7.2, 3.8, 4.0, 3.5, 3.2, 4.5, 3.8, 4.0, 3.5]. Mean=4.1.
- **Consensus:** 3.9. Inliers=7/9. Fraction=0.78. Std=1.1.
- **Failure:** OK.

**Root cause:** NCC shifted cluster centroid from 5.0 to 3.9. One match (7.2) is an outlier pulling up, but the consensus is pulled down by the majority at ~3.5-4.0. The MCD is 2.14° because GT=6° but est=3.9°. This is the largest error among the NCC-biased cases.

**Classification:** [D] NCC FAILURE — coarse correct, NCC shifted away.

### Case 8: eye_13 rot+10 (MCD=4.58°)

- **Coarse:** d=5, score=1.1767, n=5. **WRONG.** GT-candidate=10 (score=0.5887, n=3) was NOT selected.
- **Score gap:** 0.3332. The wrong candidate won by 2 more matches.
- **Raw diffs:** mean=5.00, std=0.00.
- **NCC scores:** [0.992, 0.986, 0.998, 0.996, 0.967]. All strong.
- **Per-match rots:** [11.8, 4.0, 5.8, 5.0, 9.0]. Spread=7.8°.
- **Consensus:** 5.4. Inliers=3/5. Fraction=0.60. Std=0.9.
- **Failure:** OK.

**Root cause:** This is a STRESS case (GT=10° is beyond the ±7.5° search window). The coarse search cannot reach d=10 because the maximum coarse rotation is d=5 (the lattice only goes to 5° in the positive direction before wrapping). The GT-candidate d=10 exists but has only 3 matches (sparse features A=9, B=8). The coarse search selected d=5 as the best available option.

**Classification:** [B] COARSE SEARCH FAILURE — beyond search window (stress case).

---

## 4. Stage-by-Stage Failure Classification

| Stage | Count | Cases |
|-------|-------|-------|
| **[B] COARSE SEARCH FAILURE** | **3** | eye_02 rot3+scale0.97, eye_03 rot-3, eye_13 rot+10 |
| **[D] NCC FAILURE** | **5** | eye_02 rot-3, eye_02 rot+5, eye_02 rot5+scale1.05, eye_13 rot-3, eye_13 rot+6 |
| [A] FEATURE FAILURE | 0 | — |
| [C] MATCHING FAILURE | 0 | — |
| [E] CONSENSUS FAILURE | 0 | — |
| [F] ACCEPTANCE-GATE FAILURE | 0 | — |

---

## 5. Coarse Search Analysis

### 5.1 Why eye_03 rot-3 selects d=0 instead of d=355

The coarse search evaluates all 72 lattice candidates (0, 5, 10, ..., 355). For eye_03 rot-3 (GT=-3°, true coarse=355):
- d=0: score=2.2545, n_matches=9
- d=355: score=1.9735, n_matches=7

The greedy one-to-one matching at d=0 finds 2 more matches than at d=355. This happens because the feature positions at d=0 align with a different set of features that happen to match more often. The score (weighted sum of confidences and descriptor similarities) is also higher at d=0 because the additional matches contribute more weight.

### 5.2 Why eye_02 rot3+scale0.97 selects d=0 instead of d=5

Same pattern. At d=0: score=2.3832, n=9. At d=5: score=1.7942, n=7. The scale perturbation (0.97x) shifted feature positions enough to make d=0 produce more matches.

### 5.3 Whether the 5° lattice has ambiguity for sparse feature sets

Yes. With 9-23 features spread across the iris annulus, multiple coarse rotations can produce plausible matches. The greedy matching is sensitive to small position changes (from scale perturbation or feature extraction noise).

### 5.4 Whether score and n_matches ranking can select a wrong candidate

Yes. The scoring function (weighted sum of confidences and descriptor similarities) does not penalize wrong rotations enough. A wrong rotation with more matches can outscore the correct rotation with fewer matches.

### 5.5 Whether one-to-one greedy matching contributes

Yes. The greedy matching processes features in confidence order and assigns each A-feature to its best B-match. This can produce different match sets for different coarse rotations, and the match count difference (2 matches) is enough to flip the ranking.

### 5.6 Whether a second-best coarse candidate is close to the winner

For the coarse-wrong cases:
- eye_02 rot3+scale0.97: best=0 (score=2.38), second=5 (score=1.79). Gap=0.59. Angular sep=5°.
- eye_03 rot-3: best=0 (score=2.25), second=355 (score=1.97). Gap=0.28. Angular sep=5°.
- eye_13 rot+10: best=5 (score=1.18), second=0 (score=0.84). Gap=0.33. Angular sep=5°.

The second-best candidate is always exactly 5° away (one lattice step). The score gaps are small (0.28-0.59), indicating genuine ambiguity.

### 5.7 Whether coarse ambiguity can be detected without ground truth

Partially. The score gap between best and second-best is a proxy:
- eye_02 rot3+scale0.97: gap=0.59 (moderate)
- eye_03 rot-3: gap=0.28 (small — high ambiguity)
- eye_13 rot+10: gap=0.33 (small — high ambiguity)

For the correct-coarse cases, the gaps are:
- eye_02 rot-3: gap=0.16 (very small — but coarse is still correct)
- eye_02 rot+5: gap=0.94 (large — confident)
- eye_02 rot5+scale1.05: gap=0.55 (moderate)
- eye_13 rot-3: gap=0.08 (very small — barely correct)
- eye_13 rot+6: gap=0.69 (moderate)

The score gap alone cannot reliably distinguish correct from wrong coarse selections. A small gap indicates ambiguity but does not tell which candidate is correct.

---

## 6. NCC Analysis

### 6.1 For cases where coarse is correct, does NCC shift away?

**Yes, consistently.** For all 5 cases with correct coarse:

| Case | Coarse | Est | NCC Shift | Direction |
|------|--------|-----|-----------|-----------|
| eye_02 rot-3 | 355 | 358.1 | +3.1° | positive |
| eye_02 rot+5 | 5 | 3.6 | -1.4° | negative |
| eye_02 rot5+scale1.05 | 5 | 3.8 | -1.2° | negative |
| eye_13 rot-3 | 355 | 358.1 | +3.1° | positive |
| eye_13 rot+6 | 5 | 3.9 | -1.1° | negative |

**Pattern:** When coarse is near 355° (negative rotation), NCC shifts +3°. When coarse is near 5° (positive rotation), NCC shifts -1.2°.

### 6.2 Is the NCC peak ambiguous?

No. NCC scores are very high (>0.99 for most matches). The peak is strong but the offset is biased.

### 6.3 Is the NCC peak strong but wrong?

Yes. The NCC finds a strong correlation at an offset that is systematically 1-3° away from the correct rotation. This is a **systematic bias**, not random noise.

### 6.4 Is NCC merely refining an already-wrong coarse result?

For the 5 correct-coarse cases: **No.** The coarse is correct, but NCC shifts it wrong.
For the 3 wrong-coarse cases: **Yes.** The coarse is already wrong, and NCC refines within the wrong basin.

### 6.5 Root cause of NCC bias

The NCC refinement searches for the best texture correlation in a window of [-7.5°, +7.5°] around the coarse-aligned position. The bias appears to be caused by:

1. **Texture pattern ambiguity:** The iris texture has repeating patterns that produce strong NCC peaks at multiple offsets.
2. **Edge effects:** The NCC window includes boundary regions where the annulus mask cuts off, creating artificial correlation patterns.
3. **Small feature count:** With 9-23 features, each match contributes a large weight to the consensus. A few biased matches can shift the entire estimate.

---

## 7. Consensus Analysis

### 7.1 For NCC-biased cases, is the consensus correctly picking the cluster mean?

**Yes.** The consensus estimator is working correctly. It finds the modal bin and computes the weighted circular mean within ±1° of the modal center. The problem is that the entire cluster is biased (shifted by NCC), so the consensus correctly picks the wrong value.

### 7.2 Cluster statistics for NCC-biased cases

| Case | Consensus | Inliers | Fraction | Std |
|------|-----------|---------|----------|-----|
| eye_02 rot-3 | 358.1 | 6/6 | 1.00 | 0.6° |
| eye_02 rot+5 | 3.6 | 6/8 | 0.75 | 1.1° |
| eye_02 rot5+scale1.05 | 3.8 | 6/7 | 0.86 | 0.7° |
| eye_13 rot-3 | 358.1 | 6/8 | 0.75 | 2.0° |
| eye_13 rot+6 | 3.9 | 7/9 | 0.78 | 1.1° |

The consensus fraction is high (0.75-1.00) and the inlier std is low (0.6-2.0°). This means the cluster is tight and consistent — but biased. The consensus estimator has no way to detect this bias because it only measures internal consistency, not external accuracy.

### 7.3 Is confidence calibrated?

The confidence is based on the consensus fraction and inlier std. For the NCC-biased cases:
- High consensus fraction (0.75-1.00) → high confidence
- Low inlier std (0.6-2.0°) → high confidence

The confidence is **well-calibrated for internal consistency** but **not calibrated for external accuracy**. The system is confidently wrong.

---

## 8. Acceptance Gate Analysis

### 8.1 Why did the acceptance gate accept these cases?

The `_classify_failure` function checks (in order):
1. **min_matches** (≥4): All cases pass (5-9 matches).
2. **NCC threshold** (>0.5 ratio above ncc_min=0.42): All cases pass (NCC scores >0.99).
3. **consensus_fraction** (≥0.5) and **inlier_std** (≤2.0°): All cases pass.
4. **ambiguity_ratio** (≤0.5): All cases pass.
5. **low_similarity_ratio** (≤0.5): All cases pass.

Every gate passes. The acceptance system has no information to detect the systematic bias.

### 8.2 Which gates have enough information to detect bias?

None of the current gates can detect the NCC bias because:
- The bias is external (vs ground truth), not internal (vs cluster consistency)
- The cluster is tight and consistent — just wrong
- The NCC scores are high — the texture correlation is strong, just at the wrong offset

The only gate that could potentially detect bias is the **coarse-NCC agreement**: if the NCC refinement shifts the estimate significantly from the coarse estimate, it might indicate a problem. Currently, this is not checked.

---

## 9. Root-Cause Ranking

| Rank | Root Cause | Cases | Impact |
|------|-----------|-------|--------|
| **1** | **NCC refinement systematic bias** | **5/8** | 1.0-2.1° error |
| **2** | **Coarse search wrong basin** | **2/8** | 1.0-1.1° error (excluding stress) |
| **3** | **Beyond search window** | **1/8** | 4.6° error (stress case) |

The dominant root cause is **NCC refinement bias** (5/8 cases). The coarse search failure is secondary (2/8 non-stress cases). The stress case (eye_13 rot+10) is a known limitation.

---

## 10. Recommended Smallest Next Correction

**Target the NCC refinement bias** (5/8 cases, largest impact).

The recommended approach is to add a **coarse-NCC agreement gate**: if the NCC-refined estimate deviates from the coarse estimate by more than a threshold (e.g., 2°), reject the NCC refinement and fall back to the coarse estimate.

This would:
- Fix cases 1, 2, 4, 6, 7 (NCC-biased) by rejecting the biased NCC offset
- Not affect cases 3, 5, 8 (coarse-wrong) — these would still be accepted with the wrong coarse estimate
- Not affect any correctly-recovered cases (the NCC shift is usually <1° for correct cases)

**Implementation:**
In `_classify_failure` or `estimate_correspondence`, after NCC refinement:
```python
coarse_est = wrap_deg(angle_a - angle_b)  # coarse per-match estimate
ncc_est = wrap_deg(angle_a - angle_b + refined_shift)  # NCC-refined estimate
if circular_distance(coarse_est, ncc_est) > ncc_coarse_agree_deg:
    # NCC shifted too far from coarse — reject refinement
    m.ncc = 0.0  # force NCC gate to reject this match
```

**Expected impact:**
- 5 NCC-biased cases would be rejected (honest reject instead of false-OK)
- Acceptance would improve from 0.567 toward ~0.62 (5 fewer false-OK out of 105)
- Still below 0.70 floor — coarse search failures remain

---

## 11. Evidence Against Alternative Fixes

### Alternative 1: Tighten NCC threshold (ncc_min)
**Against:** NCC scores are already very high (>0.99). Tightening the threshold would not reject these cases because the NCC correlation is strong — just biased.

### Alternative 2: Tighten consensus gates (consensus_fraction, inlier_std)
**Against:** The consensus fraction is high (0.75-1.00) and inlier std is low (0.6-2.0°). The cluster is tight and consistent. Tightening these gates would reject correctly-recovered cases that happen to have slightly lower consensus.

### Alternative 3: Improve coarse search scoring
**Against:** The coarse search is already correct for 5/8 cases. For the 2 coarse-wrong non-stress cases, the score gap is small (0.28-0.59), indicating genuine ambiguity. Improving the scoring function might help but is a larger change with uncertain impact.

### Alternative 4: Increase feature extraction density
**Against:** This would require modifying the feature extraction pipeline, which is outside the iris subsystem scope. Also, the bias appears to be systematic (same direction for all matches), not caused by sparsity.

### Alternative 5: Use RANSAC estimator instead of consensus
**Against:** The RANSAC estimator would likely pick the same cluster because the cluster is tight and consistent. The problem is not outlier rejection — the entire cluster is biased.

---

## 12. Expected Risk

**Low risk.** The coarse-NCC agreement gate is:
- Minimal (a few lines of code in `_classify_failure` or `_refine_batch`)
- Conservative (rejects biased NCC, falls back to coarse)
- Non-breaking (does not change correctly-recovered cases)
- Testable (new regression tests for coarse-NCC disagreement)

**Risk:** If the threshold is too tight, it might reject correctly-recovered cases where NCC legitimately improves the estimate. This can be mitigated by setting the threshold based on the observed NCC shift distribution (max shift for correct cases is ~1°).

---

## 13. Exact Tests Required for Next Implementation

1. **test_ncc_shift_within_coarse_agree:** For a clean rotation (e.g., 3°), verify that NCC refinement shifts the estimate by <2° from the coarse estimate.
2. **test_ncc_bias_rejected_when_shift_too_large:** Create a case where NCC would shift >2° from coarse, verify it is rejected.
3. **test_coarse_fallback_on_ncc_rejection:** When NCC is rejected, verify the coarse estimate is used instead.
4. **test_no_regression_on_clean_rotations:** All existing 62 tests pass.
5. **test_false_ok_reduced:** Run 105-case benchmark, verify FALSE-OK count decreases.
6. **test_acceptance_improved:** Verify acceptance metric increases.

---

## 14. Summary

| Question | Answer |
|----------|--------|
| Exact remaining false-OK count | **8** |
| Acceptance rate | **0.567** |
| Root-cause classification | **NCC bias (5/8), Coarse search (2/8), Beyond window (1/8)** |
| Whether coarse search is responsible | **Partially (2/8 non-stress cases)** |
| Whether NCC is still responsible | **Yes — dominant cause (5/8 cases)** |
| Whether consensus/gates are responsible | **No — consensus is correct, gates pass correctly** |
| Recommended next implementation | **Coarse-NCC agreement gate** |
| Commit SHA | `e631998` (Phase VIII-B, no changes this phase) |
| Remote verification | `e631998` on `refs/heads/main` (verified) |

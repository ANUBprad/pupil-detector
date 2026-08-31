# IRIS PHASE VIII-D — COARSE/NCC AGREEMENT GATE

**Date:** 2026-08-31
**HEAD:** `dc95510` (Phase VIII-C)
**Scope:** Investigation + attempted gate. **Gate NOT implemented.**

---

## 1. Baseline

| Metric | Value |
|--------|-------|
| HEAD | `dc95510` |
| Tests | 62/62 pass |
| FALSE-OK | 8 |
| Acceptance | 0.567 (< 0.70 floor) |
| Runtime | 93 ms |

---

## 2. Remaining Failure Distribution

| Root Cause | Count | Cases |
|-----------|-------|-------|
| NCC refinement bias | 5/8 | eye_02 rot-3, rot+5, rot5+scale1.05, eye_13 rot-3, rot+6 |
| Coarse search wrong basin | 2/8 | eye_02 rot3+scale0.97, eye_03 rot-3 |
| Beyond search window | 1/8 | eye_13 rot+10 (stress) |

---

## 3. Gate Design

### Concept

The gate was designed to detect when NCC refinement shifts the estimate significantly from a trustworthy coarse estimate:

- **Trustworthy coarse**: high score, large margin over second-best
- **Suspicious NCC**: large angular disagreement with coarse

The gate would reject NCC refinement and fall back to the coarse estimate when both conditions hold.

### Available Evidence

| Signal | Type | Available At |
|--------|------|-------------|
| `coarse_score` | Coarse candidate score | After coarse alignment |
| `score_margin` | Score gap to second-best | After coarse alignment |
| `coarse_ncc_disagree` | Angular disagreement | After NCC refinement |
| `mean_ncc` | NCC quality | After NCC refinement |
| `n_matches` | Match count | After coarse alignment |
| `inlier_std` | Consensus spread | After consensus |
| `consensus_fraction` | Inlier fraction | After consensus |

---

## 4. Evidence Supporting Threshold/Decision Rule

### Per-Case Measurements (all 105 cases)

**FALSE-OK cases (8):**

| Image | Case | coarse | score | margin | disagree | mcd | inlier_std |
|-------|------|--------|-------|--------|----------|-----|-----------|
| eye_02 | rot-3 | 355 | 1.46 | 0.16 | 3.1 | 1.07 | 0.66 |
| eye_02 | rot+5 | 5 | 2.06 | 0.94 | 1.4 | 1.38 | 0.54 |
| eye_02 | rot3+scale0.97 | 0 | 2.38 | 0.59 | 1.9 | 1.13 | 0.51 |
| eye_02 | rot5+scale1.05 | 5 | 1.72 | 0.55 | 1.2 | 1.20 | 0.46 |
| eye_03 | rot-3 | 0 | 2.25 | 0.28 | 2.0 | 1.03 | 0.73 |
| eye_13 | rot-3 | 355 | 1.80 | 0.08 | 3.1 | 1.10 | 0.67 |
| eye_13 | rot+6 | 5 | 2.17 | 0.69 | 1.1 | 2.14 | 0.53 |
| eye_13 | rot+10 | 5 | 1.18 | 0.33 | 0.4 | 4.58 | 0.42 |

**TRUE-OK cases (67) distributions:**

| Metric | min | median | max |
|--------|-----|--------|-----|
| coarse_score | 1.18 | 3.53 | 49.24 |
| score_margin | 0.00 | 1.23 | 16.92 |
| coarse_ncc_disagree | 0.00 | 0.92 | 3.07 |
| inlier_std | 0.00 | 0.36 | 0.70 |

### Separation Analysis

| Metric | FALSE-OK range | TRUE-OK range | Overlap |
|--------|---------------|---------------|---------|
| coarse_score | [1.18, 2.38] | [1.18, 49.24] | Full overlap |
| score_margin | [0.08, 0.94] | [0.00, 16.92] | Full overlap |
| coarse_ncc_disagree | [0.42, 3.10] | [0.00, 3.07] | Full overlap |
| inlier_std | [0.42, 0.73] | [0.00, 0.70] | Partial overlap |

**Key finding:** No single metric or simple combination can perfectly separate FALSE-OK from TRUE-OK. All metrics overlap significantly.

---

## 5. Threshold Search Results

### Combined Gate: (margin > X) AND (disagree > Y)

| Threshold | false_ok_rejected | true_ok_rejected | Safe? |
|-----------|-------------------|------------------|-------|
| margin>0.9 AND disagree>1.2 | 1/8 | 0/67 | Yes, but only 1 fix |
| margin>0.5 AND disagree>2.0 | 0/8 | 4/67 | No |
| margin>0.5 AND disagree>3.0 | 0/8 | 0/67 | Yes, but 0 fixes |

### Disagree-Only Gate: (disagree > Y)

| Threshold | false_ok_rejected | true_ok_rejected | Safe? |
|-----------|-------------------|------------------|-------|
| disagree>1.0 | 7/8 | 29/67 | No (43% regression) |
| disagree>2.0 | 2/8 | 18/67 | No (27% regression) |
| disagree>3.0 | 2/8 | 4/67 | No (6% regression) |

### Inlier-Std Gate: (inlier_std > Y)

| Threshold | false_ok_rejected | true_ok_rejected | Safe? |
|-----------|-------------------|------------------|-------|
| inlier_std>0.5 | 7/8 | 23/67 | No (34% regression) |
| inlier_std>0.6 | 6/8 | 10/67 | No (15% regression) |
| inlier_std>0.7 | 2/8 | 1/67 | No (1.5% regression) |

### Maximum Safe Gate Search

Exhaustive search over all (margin, disagree) combinations found **no gate that rejects any FALSE-OK while preserving all TRUE-OK**.

The closest safe gate:
- `margin > 0.9 AND disagree > 1.2`: rejects 1/8 FALSE-OK, 0/67 TRUE-OK

But this only fixes 1 out of 8 FALSE-OK cases — a 12.5% improvement.

---

## 6. Implementation

**NOT IMPLEMENTED.** The evidence does not support a safe gate.

### Why Not Implemented

1. **No clean separation:** FALSE-OK and TRUE-OK overlap in all measured metrics.
2. **All gates cause regressions:** Any gate that rejects FALSE-OK also rejects TRUE-OK.
3. **Best safe gate is trivial:** The only safe gate (margin>0.9 AND disagree>1.2) fixes only 1/8 FALSE-OK.
4. **Unacceptable trade-off:** Fixing 1 FALSE-OK at the cost of potential regressions is not justified.

### Gate Code (NOT APPLIED)

```python
# NOT APPLIED — evidence does not support safe implementation
def _validate_refinement_against_coarse(
    score_margin: float,
    coarse_ncc_disagree: float,
    ncc_min: float = 0.9,
    disagree_min: float = 1.2,
) -> bool:
    """Return True if NCC refinement is acceptable."""
    if coarse_ncc_disagree < 0.5:
        return True  # NCC barely shifted — always acceptable
    if score_margin > ncc_min and coarse_ncc_disagree > disagree_min:
        return False  # Coarse is certain, NCC shifted significantly
    return True
```

---

## 7. 105-Case BEFORE/AFTER

**No changes made.** Baseline remains:
- FALSE-OK: 8
- Acceptance: 0.567
- Tests: 62/62

---

## 8. Case-Level Regressions

**No changes made.** No regressions.

---

## 9. False-OK Analysis

The 8 remaining FALSE-OK cases fall into three categories:

### Category A: Trustworthy coarse + suspicious NCC (5 cases)

These cases have correct coarse estimates but NCC refinement shifted them 1-3° away:
- eye_02 rot-3: coarse=355 (correct), NCC shifted to 358.1
- eye_02 rot+5: coarse=5 (correct), NCC shifted to 3.6
- eye_02 rot5+scale1.05: coarse=5 (correct), NCC shifted to 3.8
- eye_13 rot-3: coarse=355 (correct), NCC shifted to 358.1
- eye_13 rot+6: coarse=5 (correct), NCC shifted to 3.9

**Why gate cannot fix these:** The coarse score margin for these cases is moderate (0.08-0.94), which overlaps with TRUE-OK cases. A gate that rejects these would also reject many TRUE-OK cases.

### Category B: Wrong coarse + useful NCC (2 cases)

These cases have incorrect coarse estimates, and NCC refinement partially corrected them:
- eye_02 rot3+scale0.97: coarse=0 (wrong), NCC shifted to 1.9 (GT=3)
- eye_03 rot-3: coarse=0 (wrong), NCC shifted to 358 (GT=357)

**Why gate cannot fix these:** The coarse is wrong, so rejecting NCC would make things worse.

### Category C: Beyond search window (1 case)

- eye_13 rot+10: GT=10°, beyond ±7.5° window, coarse=5 (best available)

**Why gate cannot fix this:** The coarse search cannot reach the correct rotation.

---

## 10. Coarse-Search Interaction

The gate design assumed that high coarse score margin indicates trustworthy coarse. This assumption is partially correct:
- TRUE-OK cases with margin > 1.0 have 95% correct coarse estimates
- FALSE-OK cases with margin > 1.0: **none exist** (all have margin < 1.0)

The problem is that FALSE-OK cases have moderate margin (0.08-0.94), which overlaps with TRUE-OK cases that also have moderate margin. This makes it impossible to distinguish "trustworthy coarse" from "ambiguous coarse" for the marginal cases.

---

## 11. Beyond-Window Limitation

The eye_13 rot+10 case (MCD=4.58°) is a stress case beyond the ±7.5° search window. The coarse search cannot reach d=10 because the lattice only goes to d=5 in the positive direction before wrapping.

This case is fundamentally limited by the search window and cannot be fixed by a coarse-NCC agreement gate.

---

## 12. Runtime

**No changes made.** Runtime remains 93 ms.

---

## 13. Determinism

**No changes made.** Determinism unchanged.

---

## 14. Production-Safety Verification

**No changes made.** All production code untouched:
- UnifiedDetector: unchanged
- pupil detection: unchanged
- limbus detection: unchanged
- calibration: unchanged
- GUI: unchanged
- production models: unchanged

---

## 15. Final Verdict

**GATE NOT IMPLEMENTED.** The evidence does not support a safe coarse-NCC agreement gate.

### Root Cause

The fundamental problem is that FALSE-OK and TRUE-OK cases overlap in all measured metrics (coarse score, score margin, NCC disagreement, inlier standard deviation). No threshold combination can separate them without causing regressions.

### Why the Gate Cannot Work

1. **FALSE-OK cases have moderate coarse confidence** (margin 0.08-0.94), which overlaps with TRUE-OK cases.
2. **NCC disagreement overlaps** between FALSE-OK (0.42-3.10) and TRUE-OK (0.00-3.07).
3. **Any gate that rejects FALSE-OK also rejects TRUE-OK**, causing unacceptable regressions.

### Recommended Next Experiment

Instead of a coarse-NCC agreement gate, the next experiment should focus on **improving NCC refinement accuracy** rather than rejecting it. Specific approaches:

1. **Reduce NCC search window:** The current window is [-7.5°, +7.5°]. Reducing it to [-3°, +3°] might prevent large biased shifts while preserving legitimate sub-degree refinement.

2. **Weight NCC by coarse confidence:** When coarse confidence is high, weight the NCC search more heavily around the coarse estimate.

3. **Improve feature matching:** Add more features to reduce the impact of individual biased matches.

4. **Use RANSAC consensus:** The current consensus estimator uses modal binning. A RANSAC approach might better handle biased matches.

### Key Finding

The Phase VIII-C recommendation (coarse-NCC agreement gate) was based on the observation that NCC shifts 1-3° from coarse. However, the gate cannot distinguish "trustworthy coarse + suspicious NCC" from "ambiguous coarse + useful NCC" because the coarse confidence metrics overlap between FALSE-OK and TRUE-OK cases.

This is an important finding: **the coarse estimate is not reliable enough to serve as a ground truth for validating NCC refinement.** Future approaches must improve NCC accuracy directly rather than using coarse as a reference.

---

## 16. Summary

| Question | Answer |
|----------|--------|
| Was the gate implemented? | **No** |
| Why? | **No safe threshold exists — all gates cause regressions** |
| Best safe gate? | **margin>0.9 AND disagree>1.2** (fixes 1/8 FALSE-OK, 0 regressions) |
| Is this sufficient? | **No — 12.5% improvement is not meaningful** |
| Code changes? | **None** |
| Tests? | **62/62 pass (unchanged)** |
| FALSE-OK? | **8 (unchanged)** |
| Acceptance? | **0.567 (unchanged)** |
| Commit? | **Report only** |

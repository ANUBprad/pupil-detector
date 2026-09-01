# Phase XX-A: Real ELITA Video Baseline Report

**Generated:** 2026-09-02 00:31:07

---

## 1. Dataset Inventory

| Property | 20250218_232912A.mp4 | 20250218_233210A.mp4 |
|---|---|---|
| Resolution | 1920x1080 | 1920x1080 |
| FPS | 60.01 | 60.01 |
| Frame Count | 6,923 | 4,857 |
| Duration | 115.4s (1.9 min) | 80.9s (1.3 min) |
| Codec | h264 | h264 |
| File Size | 373.8 MB | 243.4 MB |

> **Note:** No assumptions are made about laterality, pairing, or clinical meaning.
> These are engineering observations only.

---

## 2. Methodology

- Frames sampled strategically across each video (beginning, early-middle, middle, late-middle, end, plus boundary samples)
- Each sampled frame processed through the full pipeline: `UnifiedDetector.detect()` -> `IrisFeatureDetector.detect()`
- Timing measured per-stage with `time.perf_counter()`
- No Kalman smoothing applied (raw detection output)
- Iris detection skipped when `has_both` is False (requires both pupil and limbus ellipses)

---

## 3. Per-Video Results

### 20250218_232912A.mp4

- Frames sampled: 24
- Pupil success: 24/24 (100%)
- Limbus success: 15/24 (62%)
- Iris success (status=OK): 15/24 (62%)
- Iris skipped (missing prerequisite): 4/24 (17%)

#### Frame-by-Frame Detail

| Frame | Time | Pupil | Limbus | Iris Status | Features | Coverage | Time (ms) |
|-------|------|-------|--------|-------------|----------|----------|-----------|
| 0 | 0.0s | OK | OK | OK | 29 | 0.56% | 14055 |
| 1 | 0.0s | OK | OK | OK | 33 | 0.60% | 9488 |
| 2 | 0.0s | OK | OK | OK | 31 | 0.59% | 11075 |
| 364 | 6.1s | OK | OK | OK | 72 | 1.46% | 12958 |
| 728 | 12.1s | OK | FAIL | SKIPPED_NO_LIMBUS | - | - | 26409 |
| 1092 | 18.2s | OK | FAIL | SKIPPED_NO_LIMBUS | - | - | 28956 |
| 1457 | 24.3s | OK | FAIL | NOT_RUN | - | - | 17217 |
| 1821 | 30.3s | OK | FAIL | SKIPPED_NO_LIMBUS | - | - | 11191 |
| 2185 | 36.4s | OK | FAIL | SKIPPED_NO_LIMBUS | - | - | 12374 |
| 2550 | 42.5s | OK | FAIL | NOT_RUN | - | - | 5499 |
| 2914 | 48.6s | OK | OK | OK | 72 | 0.86% | 5270 |
| 3278 | 54.6s | OK | OK | OK | 72 | 0.84% | 4151 |
| 3643 | 60.7s | OK | OK | OK | 72 | 0.86% | 4053 |
| 4007 | 66.8s | OK | OK | OK | 72 | 0.90% | 4067 |
| 4371 | 72.8s | OK | OK | OK | 72 | 0.88% | 4673 |
| 4736 | 78.9s | OK | OK | OK | 72 | 0.90% | 4431 |
| 5100 | 85.0s | OK | OK | OK | 72 | 0.91% | 3978 |
| 5464 | 91.1s | OK | OK | OK | 72 | 0.91% | 3811 |
| 5829 | 97.1s | OK | OK | OK | 72 | 0.92% | 3781 |
| 6193 | 103.2s | OK | OK | OK | 68 | 0.87% | 3750 |
| 6557 | 109.3s | OK | OK | OK | 66 | 0.84% | 3562 |
| 6920 | 115.3s | OK | FAIL | NOT_RUN | - | - | 5119 |
| 6921 | 115.3s | OK | FAIL | NOT_RUN | - | - | 5280 |
| 6922 | 115.4s | OK | FAIL | NOT_RUN | - | - | 5510 |

### 20250218_233210A.mp4

- Frames sampled: 24
- Pupil success: 24/24 (100%)
- Limbus success: 19/24 (79%)
- Iris success (status=OK): 19/24 (79%)
- Iris skipped (missing prerequisite): 1/24 (4%)

#### Frame-by-Frame Detail

| Frame | Time | Pupil | Limbus | Iris Status | Features | Coverage | Time (ms) |
|-------|------|-------|--------|-------------|----------|----------|-----------|
| 0 | 0.0s | OK | OK | OK | 23 | 0.39% | 3271 |
| 1 | 0.0s | OK | OK | OK | 22 | 0.37% | 2405 |
| 2 | 0.0s | OK | OK | OK | 24 | 0.41% | 2685 |
| 255 | 4.2s | OK | OK | OK | 72 | 1.00% | 2828 |
| 511 | 8.5s | OK | FAIL | SKIPPED_NO_LIMBUS | - | - | 9175 |
| 766 | 12.8s | OK | FAIL | NOT_RUN | - | - | 3270 |
| 1022 | 17.0s | OK | OK | OK | 72 | 0.77% | 3014 |
| 1277 | 21.3s | OK | OK | OK | 72 | 0.72% | 2920 |
| 1533 | 25.5s | OK | OK | OK | 72 | 0.83% | 2794 |
| 1789 | 29.8s | OK | OK | OK | 72 | 0.86% | 3171 |
| 2044 | 34.1s | OK | OK | OK | 72 | 0.83% | 2937 |
| 2300 | 38.3s | OK | OK | OK | 72 | 0.87% | 2858 |
| 2555 | 42.6s | OK | OK | OK | 72 | 0.91% | 2582 |
| 2811 | 46.8s | OK | OK | OK | 72 | 0.92% | 3293 |
| 3066 | 51.1s | OK | OK | OK | 72 | 0.93% | 3354 |
| 3322 | 55.4s | OK | OK | OK | 72 | 0.74% | 2869 |
| 3578 | 59.6s | OK | OK | OK | 72 | 0.91% | 2467 |
| 3833 | 63.9s | OK | OK | OK | 72 | 0.91% | 2418 |
| 4089 | 68.1s | OK | OK | OK | 72 | 0.91% | 2372 |
| 4344 | 72.4s | OK | OK | OK | 71 | 0.91% | 2534 |
| 4600 | 76.7s | OK | OK | OK | 72 | 0.74% | 1868 |
| 4854 | 80.9s | OK | FAIL | NOT_RUN | - | - | 1982 |
| 4855 | 80.9s | OK | FAIL | NOT_RUN | - | - | 1911 |
| 4856 | 80.9s | OK | FAIL | NOT_RUN | - | - | 2089 |

---

## 4. Aggregate Detection Rates

- **Total frames analyzed:** 48
- **Pupil success:** 48/48 (100%)
- **Limbus success:** 34/48 (71%)
- **Iris success:** 34/48 (71%)
- **Iris skipped:** 5/48 (10%)

---

## 5. Iris Feature Statistics

- **Frames with iris features:** 34/48
- **Feature count range:** 22 - 72
- **Feature count mean:** 63.7
- **Feature count median:** 72.0
- **Coverage range:** 0.368% - 1.462%
- **Coverage mean:** 0.819%
- **Spatial spread (std of dist from centroid):** 17.3 px (mean), 4.9 - 39.6 px (range)
- **Iris detection time:** 570 ms (mean), 204 - 1716 ms (range)

---

## 6. Failure Breakdown

| Failure Type | Count | % of Total |
|---|---|---|
| Pupil detection failure | 0 | 0% |
| Limbus detection failure (after pupil OK) | 14 | 29% |
| Iris skipped (no pupil) | 0 | 0% |
| Iris skipped (no limbus) | 0 | 0% |
| Iris NO_ROI (pupil+limbus OK but no ROI) | 0 | 0% |
| Iris detection error | 0 | 0% |
| Iris OK | 34 | 71% |
| Other/unclassified | 0 | 0% |

---

## 7. Temporal Observations

### 20250218_232912A.mp4

- Feature counts across 15 iris-positive frames: [29, 33, 31, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 68, 66]
- Coverage values: ['0.564%', '0.605%', '0.593%', '1.462%', '0.861%', '0.845%', '0.865%', '0.895%', '0.877%', '0.896%', '0.912%', '0.911%', '0.918%', '0.870%', '0.844%']
- Feature count coefficient of variation: 0.26 (stable)

### 20250218_233210A.mp4

- Feature counts across 19 iris-positive frames: [23, 22, 24, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 71, 72]
- Coverage values: ['0.395%', '0.368%', '0.406%', '1.002%', '0.767%', '0.724%', '0.830%', '0.862%', '0.826%', '0.874%', '0.912%', '0.918%', '0.927%', '0.743%', '0.913%', '0.910%', '0.914%', '0.905%', '0.735%']
- Feature count coefficient of variation: 0.28 (stable)

---

## 8. Performance Measurements

### 20250218_232912A.mp4

- **Total pipeline time:** 8777 ms (mean), 3562 - 28956 ms (range)
- **Pupil+Limbus detection:** 8280 ms (mean), 3147 - 28956 ms (range)
- **Iris detection (when run):** 796 ms (mean), 398 - 1716 ms (range)

### 20250218_233210A.mp4

- **Total pipeline time:** 2961 ms (mean), 1868 - 9175 ms (range)
- **Pupil+Limbus detection:** 2651 ms (mean), 1664 - 9175 ms (range)
- **Iris detection (when run):** 391 ms (mean), 204 - 594 ms (range)

---

## 9. Root Cause of Empty/Weak Iris Detection

Iris detection runs successfully when pupil+limbus are both detected.
Any weakness is in upstream detection, not iris processing itself.

---

## 10. What Is Working

- Pupil detection: 48/48 success
- Limbus detection: 34/48 success
- Iris detection: 34/48 success (when prerequisites met)
- Iris feature extraction produces valid features with measurable coverage and spatial spread
- Iris detection runs in ~570 ms per frame

---

## 11. What Is Failing

- Limbus detection fails on 14/48 frames (primary iris blocker)

---

## 12. Recommended Phase XX-B Changes

Based on the baseline analysis:

1. **If limbus failure is the primary blocker:** Focus Phase XX-B on improving limbus detection
   robustness for video frames (motion blur, lighting variation, partial occlusion).

2. **If iris NO_ROI is the issue:** Adjust ROI inset parameters or add fallback ROI
   construction when the annular region is degenerate.

3. **If iris features are insufficient:** Tune feature extraction parameters
   (min_contrast, num_angles, num_radii) for real surgical video characteristics.

4. **Temporal tracking:** If iris features are stable across consecutive frames,
   implement feature tracking/matching for cyclotorsion estimation.

5. **Performance:** If pupil+limbus detection is too slow for real-time video,
   consider frame skipping or pipeline optimization.

---

> **Disclaimer:** These videos establish engineering detection behavior but do not by themselves
> establish a clinically valid cyclotorsion measurement. No assumptions are made about
> pre-dock/post-dock pairing, laterality, or ground truth.

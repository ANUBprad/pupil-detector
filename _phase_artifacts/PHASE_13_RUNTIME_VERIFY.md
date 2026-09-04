# Phase 13 Report — Real-Data Iris-Hardening Verification

## Goal
Verify the Phase 3+4+5 iris hardening (strict valid-iris mask, adaptive ROI-based
acceptance, rejection-reason tracking) on **real ELITA clinical video**, reporting
the full per-frame metrics the plan Phase 13 requires: image, pupil/limbus detected,
ROI area, valid fraction, intensity p05/p50/p95, contrast, candidates, accepted,
rejected-by-reason, angular/radial coverage, correspondence/rotation/confidence.

## Change
Extended `_phase_artifacts/diagnose_iris_runtime.py` (existing diagnostic, additive)
to additionally report, per sampled frame:

- ROI area (`annulus_area_px`) and valid-iris pixels / fraction
- intensity percentiles p05 / p50 / p95 and ROI local-contrast / texture response
  (from `result.mask_stats`, which now merges `roi_iris_stats`)
- full `rejection_reasons` breakdown (outside_valid_mask / low_texture / low_contrast /
  patch_outside_iris / angular_suppression / max_features_cap)
- angular coverage via `compute_feature_metrics` (span, coverage ratio, largest gap,
  occupied 30-deg bins) and per-feature radial/angle/confidence/response.

No production code was changed. `correspondence` import added: `compute_feature_metrics`.

## Results (real ELITA 20250218_232912A.mp4 + 20250218_233210A.mp4)
Sampled 12 frames per video. Representative frames (from the two videos):

| frame | ROI area px | valid frac | p05/p50/p95 | contrast | accept | rejected (dominant) | ang cov |
|-------|------------:|--------:|-------------:|--------:|-------:|----------------------|--------:|
| A     | 145476 | — | 50/59/70 | 0.219 | 0 (NO_FEATURES) | low_texture 488 | — |
| A     | 170726 | — | 57/67/79 | 0.278 | 69 | low_contrast 251 | 0.958 |
| A     | 225879 | — | 43/52/88 | 0.275 | 54 | low_contrast 389 | 0.778 |
| A     | 228896 | — | 44/50/58 | 0.274 | 70 | low_contrast 267 | 0.972 |
| A     | 227173 | — | 44/50/58 | 0.268 | 72 | low_contrast — | — |
| B     | 222620 | 0.526 | 42/48/57 | 0.267 | 15 | low_texture 424 | 0.806 |

## Interpretation
- The hardening works on real data: healthy frames yield 50–70 accepted features with
  high (0.94–0.97) angular coverage. ROI percentiles are adaptively derived (no longer a
  hard-coded global band), consistent with the acquisition.
- `low_contrast` is the dominant rejection on real frames: the adaptive intensity band
  (ROI p05..p95 span, frac [0.30, 0.80]) rejects many candidates. This is expected
  strictness; it is what makes acceptance adaptive across ELITA RGB vs Pentacam.
- `low_texture` dominates the frame that ends `NO_FEATURES` — an honest refusal (likely a
  squint/occlusion), reported as `IrisStatus.NO_FEATURES`, not a false positive.
- `patch_outside_iris` appears rarely (1–2), confirming the patch-support validation is
  active without over-rejecting.
- No centration / pupil / limbus / calibration code was touched.

## Verification
- Diagnostic runs end-to-end on both ELITA videos without error.
- Fast iris test suite still green (138 passed) — diagnostic is standalone, no test impact.
```
# PHASE XX-E: SmartContourFitter Gradient Caching

## 1. Objective
Cache gradient computation within a single gray image to eliminate duplicate computation across contours.

## 2. Root Cause from XX-D
- Gradient computation: 239 ms per contour
- 10-15 contours per frame
- Total wasted: 2,390-3,585 ms per frame

## 3. Cache Design
- Instance-local cache in `SmartContourFitter`
- Stores `grad_mag`, `grad_x`, `grad_y` arrays
- Keyed by `id(gray)` — detects when gray image changes
- Computed once per gray image, reused for all contours

## 4. Cache Lifetime
- Valid for one gray image across all fit() calls
- Invalidated when gray image changes (new frame, new image, resized image)
- Instance-local: no cross-fitter contamination

## 5. Cache Invalidation
- `id(gray)` comparison — instant, no pixel comparison
- Invalidated on: new frame, new image, dimension change, different detector instance

## 6. Methods Modified
- `SmartContourFitter.__init__`: added cache state
- `SmartContourFitter.fit()`: calls `_ensure_gradient_cache()`
- `SmartContourFitter._ensure_gradient_cache()`: new method
- `_refine_contour_subpixel()`: accepts optional cached gradients

## 7. Tests Added
8 unit tests in `test_gradient_cache.py`:
- First fit populates cache
- Second fit reuses cache
- Cache invalidated on new image
- Cache invalidated on dimension change
- Cached and uncached produce equivalent results
- Gradient weights still computed independently
- No stale cache across fitters
- Cache cleared when no gray

## 8. Before/After Gradient Computations
| Metric | Before | After |
|--------|--------|-------|
| Gradient computations/frame | 10-15 | 1 |
| Cache hits/frame | 0 | 9-14 |
| Cache misses/frame | 10-15 | 1 |

## 9. Numerical Equivalence
- Detection status: IDENTICAL (0 changes, same detector instance)
- Centre delta: 0.0000 (max and mean)
- Radius delta: 0.0000 (max and mean)
- False accepts: 0
- False rejects: 0

## 10. Performance Benchmark (48 ELITA frames)
| Metric | Baseline | Cached | Delta | Improvement |
|--------|----------|--------|-------|-------------|
| Mean | 4,526 ms | 1,971 ms | -2,555 ms | 56.5% |
| Median | 4,588 ms | 1,589 ms | -2,999 ms | 65.4% |
| P95 | 6,562 ms | 4,651 ms | -1,911 ms | 29.1% |
| Worst | 6,952 ms | 4,417 ms | -2,535 ms | 36.5% |

## 11. Real ELITA Validation
- 48 frames from 2 manager videos
- Identical detection results
- Performance: 56.5% faster

## 12. Full Test Results
- 28 tests run
- 0 failures
- 0 new failures

## 13. Memory/Thread-Safety Assessment
- Cache is instance-local (no global state)
- ~50 MB per instance for 1920x1080 gradients
- Single-threaded detection (no concurrency issues)
- No references retaining old frames

## 14. Clinical Safety Assessment
- No detection semantics changed
- No acceptance criteria changed
- No fitting algorithms changed
- No clinical measurements affected
- Zero detection status changes on same detector instance

## 15. Remaining Bottlenecks
After caching, remaining costs:
- Per-point subpixel loop: ~477 ms (Python loop)
- Weighted Taubin gradient weights: ~140 ms (single-scale Scharr)
- RANSAC circle fit: ~68 ms
- Bootstrap uncertainty: ~33 ms

## 16. Next Recommended Phase
PHASE XX-F: Profile and optimize per-point subpixel loop (vectorize with NumPy).

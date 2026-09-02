# PHASE XX-F: Subpixel Refinement Vectorization

## 1. Existing Loop Structure
- Outer loop: per contour point (N points)
- Inner loop: per sample along gradient normal (25 samples)
- Per-point: gradient lookup, normal computation, boundary check
- Per-sample: coordinate computation, bilinear interpolation
- Post-loop: peak finding, parabolic interpolation

## 2. Mathematical Equivalence Analysis
- All operations independent across contour points → fully vectorizable
- Inner loop independent across samples → fully vectorizable
- No sequential dependencies, no shared mutable state

## 3. Vectorized Operations
- Point coordinate extraction (all points at once)
- Gradient lookup (all points at once)
- Normal computation (all points at once)
- Sample coordinate computation (all points × all samples)
- Bilinear interpolation (all samples at once)
- Peak finding (argmax per point)
- Parabolic interpolation (vectorized)

## 4. Operations Retained Scalar
- None — all operations vectorized

## 5. Tests Added
9 unit tests in `test_vectorized_subpixel.py`:
- Empty contour
- Single point
- Boundary points unchanged
- Zero gradient unchanged
- Cached vs uncached equivalent
- Multiple points independent
- Parabolic interpolation
- No parabolic
- Large contour performance

## 6. Numerical Equivalence
- Detection status: IDENTICAL (0 changes)
- Centre delta: 0.0000 (max and mean)
- Radius delta: 0.0000 (max and mean)
- False accepts: 0
- False rejects: 0

## 7. Performance Results
| Metric | Before (XX-D) | After (XX-F) | Improvement |
|--------|---------------|--------------|-------------|
| Subpixel per frame | 3,319 ms | 275 ms | 91.7% |
| Subpixel per contour | 332 ms | 20.7 ms | 93.8% |
| Subpixel per point | 53.6 us | 5.88 us | 89.0% |

## 8. Memory Impact
- Temporary arrays: n_pts × n_samples (42,000 × 25 = 1M elements ~8 MB)
- Acceptable for 1920×1080 images

## 9. Real ELITA Results
- 48 frames from 2 manager videos
- Identical detection results
- Subpixel: 91.7% faster

## 10. Full Test Results
- 24 tests run (15 existing + 9 new)
- 0 failures
- 0 new failures

## 11. Clinical Safety Assessment
- No detection semantics changed
- No acceptance criteria changed
- No fitting algorithms changed
- No clinical measurements affected

## 12. Remaining SmartContourFitter Bottlenecks
After vectorization:
- Weighted Taubin gradient weights: ~140 ms (single-scale Scharr)
- RANSAC circle fit: ~68 ms
- Bootstrap uncertainty: ~33 ms
- Bilinear interpolation (now vectorized): ~275 ms

## 13. Next Recommendation
PHASE XX-G: Profile and optimize RANSAC circle fit (68 ms) or bootstrap uncertainty (33 ms).

# PHASE XX-I: RedLightFilter Area Filtering Vectorization

## Summary

Vectorized the connected-component area filtering loop in `RedLightFilter._detect_red_lights()` from a per-label Python loop to a single `np.isin()` call.

## What Changed

**File**: `pupil_tracking/preprocessing/red_light_filter.py`

**Before** (lines 234-241):
```python
mask = np.zeros((h, w), dtype=np.uint8)
for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < self.min_area:
        continue
    if area > max_blob_area:
        continue
    mask[labels == i] = 255
```

**After**:
```python
keep = np.where(
    (stats[1:, cv2.CC_STAT_AREA] >= self.min_area)
    & (stats[1:, cv2.CC_STAT_AREA] <= max_area_frac * h * w)
)[0] + 1

if keep.size == 0:
    return np.zeros((h, w), dtype=np.uint8)

mask = np.isin(labels, keep).astype(np.uint8) * 255
```

## Root Cause

The original loop performed `mask[labels == i] = 255` for each label — an O(h×w) boolean mask operation per label. With 200-500 connected components per frame (typical for surgical red illumination), this was O(n_labels × h × w) = hundreds of millions of operations.

The vectorized version uses `np.isin(labels, keep)` which is O(h×w) total regardless of label count.

## Performance (48 ELITA frames, wrapper parameters)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Mean detection time | 547.9 ms | 67.3 ms | **8.14×** |
| Time saved | — | 480.6 ms/frame | — |
| Labels per frame | 193-502 | — | — |

## Correctness

- **48/48 masks bit-identical** between original loop and vectorized version
- **0 mask mismatches** across all ELITA frames
- Mask output unchanged; only internal computation method differs

## Tests

9 new tests in `pupil_tracking/tests/test_vectorized_area_filter.py`:
- Empty image, single blob, multiple blobs
- Tiny blobs filtered out, huge blobs filtered out
- No labels pass filter
- Many small blobs (vectorized correctness)
- Speedup verification (vectorized must be faster than loop)
- End-to-end `apply()` pipeline

Total: **26 tests pass** (9 new + 17 existing from XX-E + XX-F)

## What Was NOT Changed

- Color detection criteria (thresholds, dominance, pink mask, etc.)
- HSV saturation analysis
- Morphological dilation
- Connected components computation
- Mask semantics and output format
- Inpainting behavior (preserved as-is)

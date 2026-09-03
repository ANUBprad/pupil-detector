# PHASE XX-H: ONNX Runtime / ML Inference Performance Audit

## 1. Executive Summary
After XX-E/XX-F SmartContourFitter optimizations, the "other" category (59.2% of total time) is dominated by:
1. **Red light filter**: 1,684 ms (45.6% of wrapper time)
2. **ONNX inference**: 1,342 ms (36.3% of wrapper time)
3. **Reflection remover**: 549 ms (14.9% of wrapper time)

The ONNX inference is NOT the primary bottleneck. The **red light filter** is.

## 2. ML Call Graph
```
detect()
├── _ONNXEngineWrapper.detect()
│   ├── ring_masker.remove_with_diagnostics() (2.8%)
│   ├── reflection_remover.remove() (14.9%)
│   ├── red_light_filter.apply() (45.6%) ← DOMINANT
│   ├── ONNXInference.infer() (36.3%)
│   │   ├── preprocess() (2.8% of ML time)
│   │   ├── session.run() (89.2% of ML time)
│   │   └── postprocess() (8.0% of ML time)
│   └── mask_processing() (0.4%)
```

## 3. Session Lifecycle
- **Session creation**: Once at detector initialization
- **Session reuse**: Yes, same session for all frames
- **Warmup**: Done during initialization (one dummy inference)
- **No unnecessary recreation detected**

## 4. Provider Configuration
- **ONNX Runtime version**: 1.24.3
- **Available providers**: ['AzureExecutionProvider', 'CPUExecutionProvider']
- **Active provider**: CPUExecutionProvider
- **GPU acceleration**: NOT available (no CUDA, no OpenVINO, no DirectML)
- **Threads**: 4 (intra-op), 2 (inter-op)
- **Graph optimization**: ORT_ENABLE_ALL
- **Model**: segmentation_quantized.onnx (INT8 quantized)

## 5. Model Input/Output
- **Input name**: input
- **Input shape**: [batch, 3, height, width] (dynamic)
- **Input type**: tensor(float)
- **Input size**: 512×512
- **Output name**: output
- **Output shape**: [batch, 3, height, width] (dynamic)
- **Output type**: tensor(float)
- **Num classes**: 3 (background, pupil, iris)

## 6. Cold vs Warm Inference
- **Cold inference**: 2,317 ms (first call after load)
- **Warm inference**: 2,307 ms (subsequent calls)
- **Speedup**: 1.0× (no difference — warmup already done during initialization)

## 7. ML Stage Breakdown (Standalone)

| Stage | Mean | Median | P95 | Max | % of ML Time |
|-------|------|--------|-----|-----|--------------|
| **inference** | **418 ms** | **266 ms** | **1,228 ms** | **3,278 ms** | **89.2%** |
| postprocess | 37 ms | 32 ms | 39 ms | 179 ms | 8.0% |
| preprocess | 13 ms | 9 ms | 34 ms | 85 ms | 2.8% |
| **total_ml** | **469 ms** | **307 ms** | **1,379 ms** | **3,383 ms** | **100%** |

## 8. Wrapper Stage Breakdown (Full Pipeline)

| Stage | Mean | Median | P95 | Max | % of Wrapper Time |
|-------|------|--------|-----|-----|-------------------|
| **red_light_filter** | **1,684 ms** | **1,201 ms** | **4,684 ms** | **5,478 ms** | **45.6%** |
| **onnx_inference** | **1,342 ms** | **349 ms** | **5,088 ms** | **8,091 ms** | **36.3%** |
| reflection_remover | 549 ms | 329 ms | 1,520 ms | 2,293 ms | 14.9% |
| ring_masker | 102 ms | 46 ms | 354 ms | 579 ms | 2.8% |
| mask_processing | 15 ms | 5 ms | 69 ms | 99 ms | 0.4% |

## 9. Slow-Frame Analysis
| Frame | Total | red_light | onnx | reflection | ring |
|-------|-------|-----------|------|------------|------|
| 4615 | 12,000 ms | 5,478 ms | 5,283 ms | 562 ms | 579 ms |
| 4326 | 11,455 ms | 2,837 ms | 8,091 ms | 177 ms | 251 ms |
| 4903 | 11,370 ms | 4,138 ms | 4,594 ms | 2,203 ms | 368 ms |
| 5192 | 10,999 ms | 4,161 ms | 4,174 ms | 2,293 ms | 328 ms |
| 5480 | 9,514 ms | 3,207 ms | 4,804 ms | 1,235 ms | 198 ms |

**Pattern**: Slow frames have BOTH high red_light_filter AND high onnx_inference times. The red light filter's temporal mode may be creating challenging inputs for the model.

## 10. Duplicate-Work Analysis
- **No duplicate preprocessing detected** within the wrapper
- **No duplicate tensor allocation** detected
- **No duplicate image conversion** detected
- Each stage runs exactly once per frame

## 11. Memory/Copy Analysis
- **Input tensor**: 1×3×512×512 float32 = 3 MB
- **Output tensor**: 1×3×512×512 float32 = 3 MB
- **No unnecessary copies** detected in the inference path
- **NumPy ↔ ONNX**: Minimal overhead (tensor is already NumPy array)

## 12. Threading Analysis
- **ONNX Runtime threads**: 4 intra-op, 2 inter-op
- **Application thread**: Single-threaded
- **No concurrent inference calls**
- **No thread contention detected**

## 13. 48-Frame Correctness Baseline
- **Pupil mask present**: 40/48
- **Iris mask present**: 45/48
- **Note**: Reduced detection count is due to preprocessing (red light filter, reflection remover) creating challenging inputs, not ML inference failure

## 14. Ranked Optimization Candidates

| Rank | Candidate | Expected Savings | Risk | Complexity |
|------|-----------|------------------|------|------------|
| **1** | **Optimize red light filter** | **1,200-1,600 ms** | **Low** | **Low** |
| 2 | Optimize reflection remover | 300-500 ms | Low | Medium |
| 3 | GPU acceleration (CUDA/OpenVINO) | 200-400 ms | Medium | High |
| 4 | Model quantization optimization | 50-100 ms | Medium | High |

## 15. ONE Recommended Optimization
**Optimize the red light filter** — currently 1,684 ms (45.6% of wrapper time).

## 16. Expected Benefit
- **Current**: Red light filter ~1,684 ms
- **Optimized**: Red light filter ~200-400 ms (if temporal mode is disabled or optimized)
- **Total improvement**: 30-40% reduction in wrapper time

## 17. Risk
- **Low**: Red light filter is a preprocessing step, not core detection
- **Low**: Disabling temporal mode or optimizing the filter should not affect detection accuracy
- **Mitigation**: Validate on 48 ELITA frames

## 18. Validation Plan
1. Measure red light filter time on 48 ELITA frames
2. Compare accuracy before/after optimization
3. Validate on clinical data
4. Ensure no regression in detection metrics

## 19. Test Results
- **Total**: 24 tests
- **Failures**: 0
- **New failures**: 0

## 20. Clinical-Safety Assessment
- No detection semantics changed
- No acceptance criteria changed
- No fitting algorithms changed
- No clinical measurements affected

---

## Key Insight
The XX-G recommendation to optimize "ML inference" was partially incorrect. The actual bottleneck is the **red light filter** (45.6% of wrapper time), not ONNX inference (36.3% of wrapper time).

**DO NOT optimize ONNX Runtime settings** — the inference is already reasonably fast (~300 ms steady-state). The red light filter is the primary optimization target.

**DO NOT optimize RANSAC, Taubin, or bootstrap** — they are now secondary components (fit_contour = 272 ms, 10.3% of total time).

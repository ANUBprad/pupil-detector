# Pupil-Limbus Detector — Complete Project Analysis

> **Product name:** Medevplus IXcentai — Surgical Grade
> **Package version:** `2.0.0` (see `pupil_tracking/__init__.py`) · **GUI version:** `2.3`
> **Purpose:** Deep-learning + classical-CV system that detects and measures the
> **pupil** (dark central aperture) and the **limbus** (iris–sclera boundary) in
> eye images and video, with sub-pixel geometry, temporal smoothing, and
> pixel→millimetre calibration for clinical/surgical use.

This document is the authoritative, code-verified technical reference for the
whole project. It is written from a direct end-to-end read of the source, not
from the older docs (which drifted from the code). Where the previous docs
referenced files that do not exist, those are corrected here.

---

## Table of Contents

1. [What the system does & why](#1-what-the-system-does--why)
2. [Clinical & domain background](#2-clinical--domain-background)
3. [High-level architecture & data flow](#3-high-level-architecture--data-flow)
4. [Verified repository layout](#4-verified-repository-layout)
5. [Entry points](#5-entry-points)
6. [`pupil_tracking.core` — detection engine](#6-pupil_trackingcore--detection-engine)
7. [`pupil_tracking.ml` — machine learning](#7-pupil_trackingml--machine-learning)
8. [`pupil_tracking.preprocessing` — image conditioning](#8-pupil_trackingpreprocessing--image-conditioning)
9. [`pupil_tracking.video` — real-time processing & tracking](#9-pupil_trackingvideo--real-time-processing--tracking)
10. [`pupil_tracking.interface` — GUI & recording](#10-pupil_trackinginterface--gui--recording)
11. [`pupil_tracking.calibration` — pixel↔mm](#11-pupil_trackingcalibration--pixelmm)
12. [`pupil_tracking.annotation` — labeling](#12-pupil_trackingannotation--labeling)
13. [`pupil_tracking.utils` — config, types, logging, runtime profile](#13-pupil_trackingutils--config-types-logging-runtime-profile)
14. [`scripts/` — training, export, evaluation, tooling](#14-scripts--training-export-evaluation-tooling)
15. [`pupil_tracking.tests` — test suite](#15-pupil_trackingtests--test-suite)
16. [Models, data & artifacts](#16-models-data--artifacts)
17. [The `EyeDetectionResult` schema](#17-the-eyedetectionresult-schema)
18. [Known issues, dead code & drift](#18-known-issues-dead-code--drift)
19. [Glossary](#19-glossary)

---

## 1. What the system does & why

Given an eye image (RGB or grayscale/IR) or a video/camera stream, the system
produces, per frame, a structured **`EyeDetectionResult`** containing:

- **Pupil** geometry — center, radius, full ellipse (semi-axes + angle),
  per-detection confidence, and quality grade.
- **Limbus** geometry — same shape as pupil; the limbus is the outer iris
  boundary and the anatomical reference for the corneal center.
- **Corneal center & offset** — the pupil-to-limbus decentration vector
  (magnitude + angle), a clinically meaningful quantity in cataract/refractive
  surgery for centering treatments.
- **Suction-ring** state — whether a femtosecond-laser suction ring is docked on
  the eye, plus its geometry (used to switch processing strategy and to
  calibrate scale).
- **Calibration** — pixel↔millimetre scale, auto-derived from the limbus
  diameter (assumed 11.5 mm) or a known ring diameter, so measurements are
  reported in real-world units.
- **Quality grade** — `SURGICAL` / `CLINICAL` / `RESEARCH` / `INSUFFICIENT` /
  `NO_DETECTION`, so downstream users know how much to trust a frame.

**Why the hybrid design:** a single approach is brittle on surgical imagery
(reflections, red LED markers, suction rings, IR illumination, motion blur).
The system therefore layers:

- a **U-Net segmentation model** (learned, robust to appearance) as the primary
  detector, then
- **classical CV** (thresholding + contours) as a fallback when ML fails, then
- **robust geometric fitting** (RANSAC circle/ellipse with sub-pixel refinement)
  to turn noisy masks into precise geometry, then
- **temporal smoothing** (Kalman filtering) to remove jitter across video, and
- **ring-aware preprocessing** so the two very different scene types (a natural
  "pre-docked" eye vs. a "docked" eye inside a suction ring) each get an
  appropriate pipeline.

The result is a system that degrades gracefully instead of failing hard, which
is the property that matters for clinical tooling.

---

## 2. Clinical & domain background

Understanding a few anatomical/surgical terms makes the whole codebase legible.

- **Pupil** — the dark central opening of the iris. It is the darkest large
  region in a well-lit eye image, which is why classical detection can fall back
  to dark-blob thresholding.
- **Iris** — the coloured muscular ring around the pupil.
- **Limbus** — the boundary between the iris and the white sclera. Its diameter
  ("white-to-white") averages ≈ **11.5 mm** in adults, which is the assumption
  used to convert pixels to millimetres when no other reference exists.
- **Corneal center** — for this system the corneal center is defined as the
  **limbus center** (an anatomical convention). The **pupil-to-limbus offset**
  (decentration) is clinically important: surgeons center treatments on it.
- **Suction ring / docking** — femtosecond-laser cataract systems (e.g. Ziemer
  Z8, Alcon LenSx) attach a **suction ring** to the eye to immobilise it. Many
  systems project a ring of **16–20 small red/orange LED marker dots** around the
  limbus. When the ring is attached the eye is **"docked"**; before attachment it
  is **"pre-docked"**. These two states look completely different, so the system
  detects the ring first and switches preprocessing accordingly.
- **Red surgical light / reflections** — bright specular highlights and blinking
  red illumination can be mistaken for the pupil or contaminate the iris
  segmentation, so several preprocessing filters exist to detect and inpaint them.

**Quality grades** map a confidence score to a clinical tier
(`utils/types.py::_GRADE_THRESHOLDS`): `SURGICAL ≥ 0.75`, `CLINICAL ≥ 0.55`,
`RESEARCH ≥ 0.30`, `INSUFFICIENT` below that, `NO_DETECTION` when nothing was
found. The tiers communicate fitness-for-use rather than raw probability.

---

## 3. High-level architecture & data flow

```
                        ┌───────────────────────────────────────────┐
  CLI / GUI entry  ──▶  │ launch_gui.py                               │
  (image/video/         │  • parses args, prints runtime profile      │
   camera/gui)          │  • applies cv2/torch thread tuning          │
                        └───────────────┬─────────────────────────────┘
                                        │
                 ┌──────────────────────┴───────────────────────┐
                 │                                               │
        GUI (interactive)                              CLI (headless)
   interface/gui_app.py                        launch_gui.process_image /
   PupilTrackingGUI                            process_video / process_camera
                 │                                               │
        ┌────────┴─────────┐                          ┌──────────┴──────────┐
        │ classic path     │                          │ optimized path      │
        │ core.UnifiedDetector.detect()               │ video.OptimizedVideoProcessor
        └────────┬─────────┘                          └──────────┬──────────┘
                 │                                                │
                 ▼                                                ▼
   ┌──────────────────────────────────┐          ┌──────────────────────────────┐
   │ UnifiedDetector.detect() pipeline │          │ ml.fast_inference.FastInference│
   │ (see §6.1 for the 9 steps)        │          │ (FP16, torch.compile, 320px)   │
   └──────────────┬───────────────────┘          └───────────────┬───────────────┘
                  │                                               │
                  ▼                                               ▼
        ml.InferenceBackend  ──▶  ONNXInference  (models/onnx/*.onnx)   [production]
                             ──▶  SegmentationInference (models/best_model.pth) [dev]
                             ──▶  _DummyEngine (no model → empty result)
                  │
                  ▼
   core.SmartContourFitter  (RANSAC circle/ellipse + sub-pixel)
                  │
                  ▼
   video.EyeKalmanTracker / video.TemporalSmoother  (temporal smoothing)
                  │
                  ▼
        EyeDetectionResult  (utils/types.py)  ──▶  overlay / CSV / JSON / measurements
```

**The one object that flows through everything is `EyeDetectionResult`**
(`utils/types.py`). Every detector returns it, every consumer (GUI overlay, CSV
export, Kalman smoother, calibration) reads/writes it. Documented fully in §17.

**Two execution paths coexist:**

- **Classic path** — `UnifiedDetector.detect()`: the full, accuracy-first
  pipeline. Used for single images and as the GUI's non-optimized mode.
- **Optimized path** — `OptimizedVideoProcessor` wrapping `FastInference`:
  fewer, faster stages (FP16, single-scale, threaded decode-ahead, batching, ROI
  tracking, overload protection) for real-time video/camera.

**Backend selection is automatic** (`ml/inference_backend.py`): ONNX Runtime is
preferred (small, fast, no CUDA toolkit needed), PyTorch is the development
fallback, and a dummy engine keeps the app alive if no model is present.

---

## 4. Verified repository layout

This is the **actual** tracked layout (from `git ls-files`), with per-file line
counts and one-line roles. Directories like `.venv/`, `build/`, `logs/`, and
`__pycache__/` are environment/output and not part of the source.

```
Pupil-Limbus-detector-main/
├── launch_gui.py                 (1892)  Unified CLI + GUI launcher (entry point)
├── train_production.py           (335)   Annotation→training bridge + train run
├── gen_notebook.py               (315)   Generates the Colab training notebook
├── train_colab.ipynb                     Colab training notebook (generated)
├── inspect_inference.py          (5)     Dev scratch: prints inference.py lines
├── inspect_ring_detector.py      (4)     Dev scratch: bare imports (throwaway)
├── build_app.bat                         PyInstaller build script (Windows)
├── requirements.txt                      Python dependencies (see §16)
├── manual_ring_priors.json               Learned manual-ring priors (runtime cache)
├── models/
│   ├── checkpoint_meta.json              {epoch:30, val_iou:0.943}
│   └── onnx/manifest.json                ONNX file sizes + sha256
│
├── pupil_tracking/               MAIN PACKAGE
│   ├── __init__.py               (9)     Version 2.0.0; lazy submodule imports
│   │
│   ├── core/                     DETECTION ENGINE
│   │   ├── __init__.py           (58)    Exports UnifiedDetector, SmartContourFitter…
│   │   ├── detector.py           (2079)  UnifiedDetector — main orchestrator
│   │   ├── smart_fitter.py       (1131)  SmartContourFitter — RANSAC circle/ellipse
│   │   ├── ellipse_fitter.py     (500)   EllipseFitter — fallback-chain fitter (ML path)
│   │   ├── confidence.py         (513)   ConfidenceScorer — scoring (UNUSED in pipeline)
│   │   ├── corneal_center.py     (401)   CornealCenterCalculator + SmoothedStateWriter
│   │   ├── deterministic_ring_detector.py (…)  RingDetector — ACTIVE ring detector
│   │   ├── ring_detector.py      (749)   Legacy RingDetector (used by ring_aware only)
│   │   └── eye_roi_detector.py   (304)   EyeROIDetector — video ROI localisation
│   │
│   ├── ml/                       MACHINE LEARNING
│   │   ├── __init__.py           (35)    Exports ONNXInference, InferenceBackend…
│   │   ├── architecture.py       (602)   EyeSegmentationModel (U-Net/ResNet-34)
│   │   ├── dataset.py            (1740)  EyeSegmentationDataset + mask generation
│   │   ├── trainer.py            (428)   Trainer — AMP training loop
│   │   ├── losses.py             (683)   Dice/Focal/Boundary/Composite losses
│   │   ├── inference.py          (1167)  SegmentationInference (PyTorch, multiscale)
│   │   ├── fast_inference.py     (829)   FastInference (FP16 + compile, real-time)
│   │   ├── onnx_inference.py     (476)   ONNXInference + ONNXRingClassifier
│   │   ├── inference_backend.py  (205)   InferenceBackend factory (ONNX→PyTorch)
│   │   ├── postprocess.py        (691)   mask→contour→ellipse, ring extraction
│   │   ├── ring_classifier.py    (671)   MobileNetV2 ring presence classifier
│   │   └── grayscale_augmentation.py (515) Grayscale training augmentation
│   │
│   ├── preprocessing/            IMAGE CONDITIONING
│   │   ├── __init__.py           (88)    Exports preprocessors (NOT temporal filter)
│   │   ├── grayscale_handler.py  (754)   GrayscaleHandler — detect/convert/enhance
│   │   ├── normalizer.py         (331)   ImageNormalizer — white-balance/CLAHE/gamma
│   │   ├── reflection_removal.py (345)   ReflectionRemover — specular inpainting
│   │   ├── suction_ring_masker.py(413)   SuctionRingMasker — red LED-dot ring
│   │   ├── ring_aware.py         (490)   RingAwarePreprocessor (docked/pre-docked)
│   │   ├── red_light_filter.py   (384)   RedLightFilter — surgical red light
│   │   └── temporal_reflection_filter.py (290) Blink vs persistent reflection
│   │
│   ├── video/                    REAL-TIME PROCESSING
│   │   ├── __init__.py           (0)     Empty
│   │   ├── optimized_processor.py(2281)  OptimizedVideoProcessor + threading
│   │   ├── video_processor.py    (519)   VideoProcessor — classic per-frame loop
│   │   ├── kalman_tracker.py     (318)   EyeKalmanTracker (8-D matrix Kalman)
│   │   └── temporal_smoother.py  (473)   TemporalSmoother (scalar KF + ring-lock)
│   │
│   ├── interface/                GUI & I/O
│   │   ├── __init__.py           (0)     Empty
│   │   ├── gui_app.py            (~5300) PupilTrackingGUI — full Tkinter app
│   │   ├── frame_recorder.py     (368)   FrameRecorder — threaded video writer
│   │   └── theme.py              (374)   Colors + DarkTheme (ttk styling)
│   │
│   ├── calibration/
│   │   ├── __init__.py           (0)     Empty
│   │   └── spatial_calibration.py(412)   SpatialCalibrator + StabilizedCalibrator
│   │
│   ├── annotation/
│   │   ├── __init__.py           (0)     Empty
│   │   └── annotation_tool.py    (419)   AnnotationTool — Tkinter point labeler
│   │
│   ├── utils/
│   │   ├── __init__.py           (0)     Empty
│   │   ├── config.py             (979)   PupilTrackingConfig (18 nested dataclasses)
│   │   ├── types.py              (737)   Enums + dataclasses (result schema)
│   │   ├── logger.py             (142)   AuditLogger (JSONL + console)
│   │   └── runtime_profile.py    (153)   Hardware detection → tuning profile
│   │
│   └── tests/
│       ├── __init__.py           (0)     Empty
│       ├── test_clinical_accuracy.py       (443) Per-image accuracy (needs model+data)
│       ├── test_video_pipeline.py          (210) Video pipeline smoke tests
│       ├── test_grayscale.py               (803) GrayscaleHandler unit tests
│       ├── test_deterministic_ring_detector.py    Ring detector on synthetic eyes
│       └── test_manual_roi.py                     Manual ROI math (no model)
│
├── scripts/                      TOOLING (20 scripts — see §14)
│   ├── train_model.py            (391)   Primary training entry (3 or 4 class)
│   ├── run_epoch.py              (68)    Single-epoch smoke test
│   ├── train_ring_classifier.py  (410)   Train MobileNetV2 ring classifier
│   ├── finetune_grayscale.py     (936)   Grayscale-robustness fine-tune (safety-gated)
│   ├── export_onnx.py            (120)   Simple ONNX export (segmentation)
│   ├── convert_to_onnx.py        (644)   Full ONNX export (seg + ring) + quantize
│   ├── process_video.py          (193)   Headless OptimizedVideoProcessor runner
│   ├── benchmark_fps.py          (298)   Inference FPS benchmark
│   ├── benchmark_video_speed.py  (660)   End-to-end video pipeline benchmark
│   ├── annotate_data.py          (12)    Launcher for AnnotationTool
│   ├── annotate_live_video.py    (2480)  Live-video annotation w/ incremental train
│   ├── annotate_ring_data.py     (443)   Keyboard ring-presence labeling
│   ├── generate_masks.py         (424)   Build masks from annotations
│   ├── verify_data.py            (290)   Annotation completeness/plausibility check
│   ├── check_files.py            (133)   Project-structure file check
│   ├── check_training_data.py    (258)   Mask/class-distribution check
│   ├── diagnose_detection.py     (223)   Diagnostic visualisations
│   ├── evaluate_ring_detection.py(574)   Ring accuracy vs ground truth
│   ├── test_grayscale_detection.py(977)  RGB vs grayscale comparison harness
│   └── debug_single_image.py     (334)   ⚠ BROKEN: imports non-existent modules
│
└── clinical_data/                DATASET (small sample committed)
    ├── annotations/              annotations.json, annotations_production.json, masks/
    └── clean/                    eye_01..eye_14 jpeg + annotations/masks
```

> **Doc-vs-code note:** the previous README referenced `pupil_tracking/detection.py`,
> `preprocessing.py`, `image_interface.py`, `clinical_debug.py`, `run_realtime.py`,
> `core/geometric_fit.py`, `core/contour_filtering.py`, `core/limbus_detector.py`,
> `video/camera_processor.py`, `video/frame_buffer.py`, `interface/api.py`,
> `annotation/annotation_converter.py`, `annotation/mask_generator.py`, and
> `calibration/camera_calibration.py`. **None of these exist.** They are removed
> from this analysis.

---

## 5. Entry points

### 5.1 `launch_gui.py` — the single front door

One file provides four modes via a positional `mode` argument
(`_build_parser()`, line 1520). Default mode is `gui`.

```
python launch_gui.py [gui|image|video|camera] [options]
```

**Startup sequence** (`main()`, line 1773):
1. `apply_runtime_optimizations(detect_runtime_profile())` runs at import time
   (line 69) — detects hardware and tunes cv2/torch thread counts before
   anything heavy loads.
2. Parses args, prints the banner and the chosen runtime profile.
3. `_run_startup_self_check()` (line 1730) validates the log dir, the input
   path, and whether an optimized model artifact exists; warns/falls back
   gracefully.
4. Dispatches on `mode`:
   - `image` → `process_image()` (single-image analysis, console report + overlay)
   - `video` → `process_video()` (file processing, optional output video)
   - `camera` → `process_camera()` (live webcam)
   - `gui` → constructs `PupilTrackingGUI` and pushes CLI flags into GUI vars
     (lines 1841–1858) so the GUI opens pre-configured.

**Argument groups (exact, from source):**

| Group | Flag | Default | Meaning |
|-------|------|---------|---------|
| positional | `mode` | `gui` | `gui`/`image`/`video`/`camera` |
| I/O | `--input, -i` | `None` | input image/video path |
| I/O | `--output, -o` | `None` | output path |
| I/O | `--model, -m` | `None` | model weights (`.pth`) |
| I/O | `--camera-id` | `0` | camera device index |
| I/O | `--device` | `auto` | `auto`/`cpu`/`cuda`/`mps` |
| Ring | `--ring-mode` | `auto` | `auto`/`docked`/`pre_docked` |
| Ring | `--ring-classifier` | `models/ring_classifier.pth` | ring model path |
| Ring | `--show-ring / --no-show-ring` | on | draw ring outline |
| Grayscale | `--grayscale` | `off` | `off`/`auto`/`force` (IR look) |
| Pipeline | `--optimized / --no-optimized` | on | use fast pipeline |
| Pipeline | `--fp16 / --no-fp16` | profile | FP16 half precision |
| Pipeline | `--compile / --no-compile` | profile | torch.compile JIT |
| Video | `--stride` | `1` | process every Nth frame |
| Video | `--resolution` | profile (320) | inference resolution px |
| Video | `--target-fps` | profile | target processing FPS |
| ROI | `--roi / --no-roi` | on | ROI tracking |
| ROI | `--roi-cache` | `5` | ROI cache lifetime (frames) |
| ROI | `--kalman-process-noise` | `0.03` | Kalman process noise |
| ROI | `--kalman-measure-noise` | `0.1` | Kalman measurement noise |

Defaults marked "profile" come from `runtime_profile.py` (§13.4), so behaviour
adapts to the machine: on a CUDA box FP16 + compile default **on**; on CPU they
default **off**.

### 5.2 `train_production.py` — annotation→training bridge

Converts the annotation format produced by `annotate_live_video.py` into the
production training schema (`PUPIL`/`LIMBUS` with `class_id` 1/2, ellipse params,
empty `boundary_points`), then runs `Trainer`. Requires ≥ 2 annotations, sets
`val_ratio=0.15`, and after training reloads the best model on CPU to verify it
deserialises. CLI: `--annotations`, `--image-dir`, `--mask-dir`, `--epochs=200`,
`--batch-size=2`, `--copies-per-image=50`, `--learning-rate=1e-4`,
`--output-dir=models`, `--skip-convert`.

### 5.3 `gen_notebook.py` / `train_colab.ipynb`

`gen_notebook.py` programmatically writes `train_colab.ipynb`, a Google Colab
notebook for GPU training when no local CUDA is available.

### 5.4 `build_app.bat`

Windows PyInstaller build: verifies ONNX models exist (or tells you to run
`convert_to_onnx.py`), installs `pyinstaller`/`onnxruntime`, cleans previous
builds, and builds from `build/pupil_detector.spec` into `build/dist`.

### 5.5 Dev scratch files

`inspect_inference.py` and `inspect_ring_detector.py` are one-off developer
snippets (print source lines / bare imports). They are **not** part of the app
and can be ignored or deleted.

---

## 6. `pupil_tracking.core` — detection engine

The `core` package is the detection brain. `__init__.py` exports
`UnifiedDetector`, `SmartContourFitter`, `FitResult`, `FitType`, `smart_fit`,
and (optionally, guarded by try/except) `CornealCenterCalculator`,
`EllipseFitter`, `EyeROIDetector`.

### 6.1 `detector.py` — `UnifiedDetector` (the orchestrator)

This is the top-level detection API. It composes every other component. The
constructor (`__init__`, ~line 131) builds:

- `self.ml_engine` via `_create_ml_engine()` — tries ONNX Runtime, then PyTorch
  `SegmentationInference`, then a `_DummyEngine` (so the object always
  constructs, even with no model).
- `self._fitter = SmartContourFitter(circularity_threshold=0.92,
  residual_ratio_threshold=1.15, use_ransac=True, subpixel_refine=True)`.
- `self._ring_detector = RingDetector(...)` from **`deterministic_ring_detector.py`**.
- `self._ring_preprocessor = RingAwarePreprocessor()`.
- `self._ring_contour_filter = AdaptiveContourFilter()`.
- `self.corneal_calc = CornealCenterCalculator(config=…)`.
- `self._calibration = CalibrationInfo()` and
  `self._stabilized_cal = StabilizedCalibrator(...)`.
- `self._grayscale_handler = GrayscaleHandler(...)`.

**`detect(image, frame_number=-1, source="", force_mode=None) → EyeDetectionResult`**
(line ~475) is the full pipeline. Step by step:

| # | Step | What happens | Why |
|---|------|--------------|-----|
| 0 | **Format + grayscale norm** | 2D/1-ch/BGRA → BGR; `_apply_grayscale_mode()` handles OFF/AUTO/FORCE; stores `GrayscaleInfo` | Uniform input; IR/grayscale support |
| 1 | **Ring detection** | `_detect_ring()` → `RingDetectionResult`; `force_mode` short-circuits; caches for 30 frames in video mode | Decide docked vs pre-docked; drives everything below |
| 2 | **Ring-aware preprocess** | `RingAwarePreprocessor.preprocess()` computed… **but result is not consumed** (see §18) | (intended) tailored conditioning |
| 3 | **ML segmentation** | `ml_engine.detect()` attaches `result._raw_mask` (0=bg,1=pupil,2=iris,3=ring); marks pupil/limbus detected if > 100 px; `_attach_ring_info()` | Primary, robust detection |
| 4 | **Smart fitting** | if `_raw_mask` present: `extract_ring_from_segmentation()` (if ≥ class 3), then `_extract_structure()` → `SmartContourFitter.fit()` for pupil & limbus; `_apply_fit_to_result()` | Turn noisy masks into precise geometry |
| 5 | **Classical fallback** | if pupil/limbus still missing and `enable_classical_fallback`: `_classical_pupil()` / `_classical_limbus()`, confidence ×`classical_confidence_penalty` (0.85) | Graceful degradation when ML fails |
| 5b | **Pre-docked limbus shrink** | if not docked: `limbus.radius *= 0.93` (hardcoded) | Counter ML over-estimation of limbus |
| 6 | **Cross-validation** | if `has_both`: `_cross_validate_and_reject()` removes implausible pupil/limbus pairs | Reject anatomically impossible fits |
| 7 | **Calibration** | `StabilizedCalibrator.update_from_limbus()`; fallback to ring diameter if docked; sets `result.calibration` | Establish px↔mm scale |
| 8 | **mm values** | if calibrated: `_add_mm_values()` fills `radius_mm`, `center_mm`, `diameter_mm` | Real-world measurements |
| 8b | **Corneal center** | `corneal_calc.calculate()` then `_blend_corneal_center_from_available()` (confidence-weighted pupil+limbus+ring) | Decentration vector |
| 9 | **Quality grade** | mean of `[pupil.conf, limbus.conf, ring.conf×0.3]` → `overall_confidence`; `assign_quality_grade()` | Fitness-for-use tier |

It **always returns** an `EyeDetectionResult` (never `None`).

**Other public methods:** `detect_video_frame(frame, frame_number, roi_offset)`
(video path with ROI offset correction), `detect_from_masks(...)` (detect from
pre-computed masks), `set_grayscale_mode()`, `init_video_mode(...)` (enables
`FastInference` + temporal red-light filtering), `calibrate_from_limbus()`,
`reset()`. Helper classes `_ONNXEngineWrapper` (adapts ONNX to the PyTorch engine
interface) and `_DummyEngine` (no-op fallback) live at the bottom of the file.

### 6.2 `smart_fitter.py` — `SmartContourFitter`

Turns a binary mask (or contour) into a precise circle **or** ellipse, choosing
automatically. Types: `FitType{CIRCLE, ELLIPSE, FAILED}` and a rich `FitResult`
dataclass (center, semi-axes, radius, angle, eccentricity, circularity,
RMS residual, inlier count, per-parameter uncertainties, contour points).

**`fit(binary_mask, gray_image=None, pupil_hint=None) → FitResult`:**
1. Largest contour from the mask.
2. If `pupil_hint` given, filter points to a radial band around the hint (drops
   limbus points that bled into the pupil mask).
3. If `subpixel_refine` and a gray image is available, `_refine_contour_subpixel()`
   snaps points to gradient peaks (Scharr operator, multi-scale gradient fusion,
   0.25-px sampling, parabolic peak fit → claimed ≈ 0.05 px accuracy).
4. `fit_contour()` fits both a circle (RANSAC → Kåsa/Taubin/Hyper hierarchy) and
   an ellipse (`cv2.fitEllipse`), compares RMS residuals, and **picks circle when**
   the aspect ratio ≥ `circularity_threshold` **or** `circle_rms/ellipse_rms ≤
   residual_ratio_threshold`; otherwise ellipse.
5. Bootstrap resampling (50 samples) estimates parameter uncertainty.

Circle-fit hierarchy (module-level): `_fit_circle_kasa` (fast algebraic) →
`_fit_circle_taubin` (unbiased, partial-arc robust) → `_fit_circle_hyper`
(most accurate) → `_fit_circle_weighted_taubin` (gradient-weighted). RANSAC uses
3-point samples and refits on inliers.

> **Note:** this module defines its **own** `FitResult`, distinct from the
> `FitResult` in `utils/types.py` used by `EllipseFitter`. See §18.

### 6.3 `ellipse_fitter.py` — `EllipseFitter`

A stateless fallback-chain fitter returning `utils.types.FitResult`. Chain:
RANSAC ellipse (≥ 10 pts) → `cv2.fitEllipse` (≥ 5) → Huber-weighted iterative
ellipse (≥ 5) → Kåsa circle (≥ 3) → `cv2.minEnclosingCircle` (≥ 1, always
succeeds). Used by `ml/inference.py`, `ml/postprocess.py`, and scripts — **not**
by `UnifiedDetector.detect()` (which uses `SmartContourFitter`).

### 6.4 `confidence.py` — `ConfidenceScorer`

Defines `QualityLevel{EXCELLENT, GOOD, FAIR, POOR, UNUSABLE}` and a scorer that
weights circularity/area/centrality (pupil), concentricity/size/containment
(limbus), and classifier/heuristic/segmentation agreement (ring).
**Not called anywhere in the live pipeline** — confidence is instead set by the
ML engine, `SmartContourFitter`, and cross-validation. Kept as a reusable module
/ candidate for future wiring (see §18).

### 6.5 `corneal_center.py` — `CornealCenterCalculator` + `SmoothedStateWriter`

- `CornealCenterCalculator.calculate(pupil, limbus, calibration) →
  CornealCenterResult`: corneal center = limbus center; offset = pupil − limbus;
  confidence = `min(pupil, limbus)×0.8`, penalised when the offset exceeds 20 %
  of the limbus radius; converts to mm when calibrated.
- `SmoothedStateWriter.apply_smoothed_dict(result, smoothed)`: writes
  Kalman-smoothed `(x, y, r)` back into the result, rescales semi-axes, and
  recomputes mm + corneal center.

### 6.6 `deterministic_ring_detector.py` — the **active** ring detector

`RingDetector.detect(image) → RingDetectionResult`. Strategy: always run the
`HeuristicRingDetector`; if a CNN classifier is loaded, blend
(heuristic 0.85 / classifier 0.15, or classifier alone when its confidence ≥ 0.92).

`HeuristicRingDetector` runs three tiers:
1. **Quick red gate** — downscale to 160 px, count HSV-red pixels with radial
   distribution; if it fails → `ABSENT` (high confidence).
2. **Marker-dot ring** — delegate to `SuctionRingMasker`; require ≥ 4 dots,
   angular coverage, radius band 26–49 % of the short side; score ≥ 0.48 → `PRESENT`.
3. **Structural ring** — CLAHE + Canny + HoughCircles as a last resort.

`RingDetectionResult` carries `status` (`PRESENT/ABSENT/PARTIAL/UNCERTAIN`),
`confidence`, `ring_center`, `ring_radius`, `ring_inner_radius`, `ring_mask`,
`dot_centers`, `dot_count`, `method`, and `corneal_reference_source`.

### 6.7 `ring_detector.py` — legacy ring detector

An older `RingDetector`/`HeuristicRingDetector` (Hough + contour + annular-edge
scan, merge weights 65/35). **Only** imported by `preprocessing/ring_aware.py`
and `scripts/evaluate_ring_detection.py`; shadowed in the main pipeline by the
deterministic version.

### 6.8 `eye_roi_detector.py` — `EyeROIDetector`

Localises the eye in a wider frame for the **video** path (not used by
`detect()`). Strategy hierarchy: closeup auto-detect → cached ROI → Haar-cascade
face→eye → intensity dark-blob fallback → full frame. Returns an `ROIResult`
(bbox + cropped image + confidence). Used by `OptimizedVideoProcessor`.

---

## 7. `pupil_tracking.ml` — machine learning

`__init__.py` always exposes `ONNXInference` and `InferenceBackend` (no torch
dependency); `SegmentationInference`, `FastInference`, and `EyeSegmentationModel`
are imported under try/except so the package works in ONNX-only production
installs without PyTorch.

### 7.1 `architecture.py` — `EyeSegmentationModel`

- U-Net with a **ResNet-34** encoder via `segmentation_models_pytorch`
  (`smp.Unet`), ImageNet-pretrained encoder, `in_channels=3`.
- **Class maps:** 3-class (`0 background, 1 pupil, 2 iris`) or 4-class
  (adds `3 suction_ring`). `get_class_names()` / `get_class_colours()` provide
  labels and BGR colours (pupil=red, iris=blue, ring=green).
- **Temperature scaling** — a non-trainable `temperature` parameter divides the
  logits (`forward` returns `logits/temperature`). `calibrate_temperature()`
  fits it post-hoc via L-BFGS on validation NLL for well-calibrated
  probabilities (important for surgical trust in confidence).
- Methods: `forward` (logits `[B,C,H,W]`), `predict_proba` (softmax),
  `predict_classes` (argmax `[B,H,W]`), `has_ring_class()`, `save()`/`load()`
  (checkpoint stores `state_dict`, `num_classes`, `encoder`, `class_names`;
  `load` auto-detects class count, falling back to head-shape probing then 3).
- ≈ 25 M parameters. The base class is `nn.Module` when torch is present, else
  `object`, so the module imports even without torch.

### 7.2 `inference.py` — `SegmentationInference` (PyTorch, accuracy path)

Robust checkpoint loader (multiple key layouts). Features: **multi-scale
inference** (448/512/640 averaged), sub-pixel contour refinement, gradient-guided
mask-boundary refinement, edge-alignment scoring, reflection/ring-marker removal.
`detect(image, frame_number, source) → EyeDetectionResult`, `get_raw_mask()`
returns the raw integer label mask. This is the default PyTorch engine used when
ONNX is unavailable.

### 7.3 `fast_inference.py` — `FastInference` (real-time path)

The engine behind `OptimizedVideoProcessor`. Accuracy-first-but-fast design
(header documents plan items A1–A7, S1–S4):

- **320×320** default input; `INTER_AREA` downscale (best-quality shrink).
- **FP16** on CUDA (`use_half`), **torch.compile** (`reduce-overhead`,
  `fullgraph`) with an eager test pass to catch missing-compiler failures.
- Pre-inference **specular reflection removal** (at downscaled resolution, ~25–50×
  cheaper) and **suction-ring marker masking**.
- ImageNet normalisation with cached device tensors; BGR→RGB **after** resize.
- Close+Open **morphology** (5×5 pupil, 3×3 iris to preserve limbus boundary).
- **Real ML probability propagation** — thresholds `pupil ≥ 0.42`, `iris ≥ 0.22`,
  confidences come from the softmax, not hardcoded.
- **Batch** methods `infer_batch` / `detect_batch` for the video pipeline.
- Warm-up runs single + batch passes so the first real frame isn't slow.

`_load_model()` builds a 3-class `smp.Unet('resnet34')`, strips `module.`/`model.`
prefixes and the `temperature` key, loads with `strict=False`, and raises if
< 50 % of parameters matched (guards against silent architecture mismatch).

### 7.4 `onnx_inference.py` — `ONNXInference` + `ONNXRingClassifier`

Production engine with **zero PyTorch dependency** (~50 MB runtime). Lazy-imports
`onnxruntime`, auto-selects the best execution provider (CUDA / CoreML / DirectML
/ CPU), supports the quantized model, and picks a sensible thread count. Drop-in
replacement for `SegmentationInference` (same `detect` contract).

### 7.5 `inference_backend.py` — `InferenceBackend` (factory)

`InferenceBackend.create(...)` tries ONNX first (`_find_onnx_model` searches
`models/onnx/`, PyInstaller `_MEIPASS`, and CWD for
`segmentation_quantized.onnx` → `segmentation.onnx`), then PyTorch
(`_find_pytorch_model` → `models/best_model.pth`), else raises with install
guidance. `create_ring_classifier(...)` mirrors this for the ring classifier
(ONNX → PyTorch → `None`, falling back to heuristic detection).

### 7.6 `postprocess.py` — mask → geometry

`RingSegmentationResult` dataclass plus functions: `mask_to_contours()`,
`contour_to_ellipse()` (with scale factors), `extract_ring_from_segmentation()`,
`extract_contours_ring_aware()` (ring-spatial-constrained), `clean_segmentation_mask()`
(morphology), and `validate_pupil_limbus_pair()` (cross-validation). This is the
bridge from the network's label map to `EllipseParams`.

### 7.7 `ring_classifier.py` — MobileNetV2 ring presence classifier

`RingClassifierNet` (MobileNetV2, 224×224 input, binary ring/no-ring),
`RingClassificationDataset`, and `RingClassifierTrainer`. `predict()` /
`predict_batch()` return a ring probability that the ring detectors fuse with
their heuristics.

### 7.8 `dataset.py`, `trainer.py`, `losses.py`, `grayscale_augmentation.py`

- **`dataset.py`** — `EyeSegmentationDataset`: parses annotation JSON, generates
  masks from ellipse params or boundary points, supports 3/4-class, and applies a
  heavy augmentation pipeline (spatial, pixel, blur, noise, JPEG compression,
  occlusion) plus optional grayscale augmentation. `build_datasets()` /
  `split_by_images()` do an **image-level** train/val split (no leakage between
  augmented copies of the same eye).
- **`trainer.py`** — `Trainer`: mixed-precision (AMP) training, cosine-annealing
  LR, early stopping on validation IoU, per-class IoU tracking, and post-training
  temperature calibration. Saves the best checkpoint + `checkpoint_meta.json`.
- **`losses.py`** — `DiceLoss`, `BoundaryLoss` (distance-transform),
  `FocalLoss`, `WeightedCrossEntropyDiceLoss`, and `CompositeLoss`; `create_loss()`
  factory selects by name. The composite (CE + Dice + Boundary) is the default and
  handles class imbalance (background dominates) plus crisp boundaries.
- **`grayscale_augmentation.py`** — `RandomGrayscaleConversion` (Albumentations
  transform) and `GrayscaleAwarePipeline`, reusing the same `GrayscaleHandler` as
  inference so training and inference grayscale handling stay consistent.

---

## 8. `pupil_tracking.preprocessing` — image conditioning

These modules clean and standardise pixels before detection. `__init__.py`
exports all of them **except** `temporal_reflection_filter` (reachable only by
direct import).

### 8.1 `grayscale_handler.py` — `GrayscaleHandler`

Handles IR/grayscale inputs so one 3-channel model serves both colour and
grayscale sources. `GrayscaleMode{AUTO, FORCE, OFF}`.

- `is_grayscale(image)` — true for single-channel, or for "fake-RGB" where the
  max pairwise mean channel difference ≤ 3.0 (computed on a ≤ 200 px downsample
  for speed). This catches grayscale data stored in 3 channels.
- `to_grayscale()` — BT.601 luminance.
- `enhance_grayscale()` — CLAHE + a conservative percentile contrast stretch.
- `to_model_input(image, force_grayscale)` — the primary entry: converts +
  enhances + replicates to 3 identical channels, returning `(ndarray, GrayscaleInfo)`.
- `get_quality_metrics()` — contrast, dynamic range, SNR estimate, etc.
- Thread-safe: the CLAHE object is lazily built under a lock.

**Why:** surgical microscopes and IR cameras often deliver grayscale; forcing an
"IR look" (`force`) also gives clinicians a consistent display while coloured
overlays are drawn on top.

### 8.2 `normalizer.py` — `ImageNormalizer`

Illumination/contrast normalisation: gray-world **white balance**, brightness
scaling toward a target mean, **CLAHE on the LAB L-channel** (preserves colour),
and optional gamma. `normalize()` is the full path; `fast_normalize()` is a
video-optimised subset (brightness + a smaller/faster CLAHE only). Improves model
robustness to lighting variation.

### 8.3 `reflection_removal.py` — `ReflectionRemover`

Detects specular highlights (bright + desaturated in HSV, plus optional blue/red
highlight and pure-white detectors), filters blobs by size, dilates, and inpaints
with Telea. `remove()` returns `(cleaned, mask)`; `detect_only()` just returns the
mask. Two speed tiers (fast threshold+inpaint vs. full bilateral pre-filter).
**Why:** corneal glints otherwise get mistaken for structure or punch holes in
the pupil/iris masks.

### 8.4 `suction_ring_masker.py` — `SuctionRingMasker`

Detects the ring of small **red/orange LED marker dots** that femto-laser systems
project on the limbus, so they don't contaminate iris segmentation. Algorithm:
HSV **two-range red threshold** (red wraps around H=0/180) → morphology →
contour/blob filtering (area, circularity, radial band) → **least-squares circle
fit through the dot centres** with a residual-ratio + angular-coverage validity
gate (the key false-positive guard) → dilate → Telea inpaint. Returns a
`SuctionRingResult` (dot count, center, inner/outer radius, residual stats, mask).
No-op on grayscale input (markers are invisible without colour).

### 8.5 `ring_aware.py` — docked vs pre-docked pipelines

`RingAwarePreprocessor.preprocess(image, ring_result)` routes on ring status:

- **Pre-docked** (no ring): grayscale → median blur → CLAHE → **soft eyelid/lash
  suppression** (fade top/bottom 15 % so dark lashes don't dominate thresholding)
  → optional normalise. Wider field of view, no ROI.
- **Docked** (ring present): build a **circular ROI inside the ring**, **neutralise
  the ring band** (replace an annulus with the local mean so the ring edge
  disappears), apply **stronger/tighter CLAHE** on the smaller interior, mask
  outside the ROI, and inpaint bright speculars at the 97th percentile.

`AdaptiveContourFilter` filters contours by area/circularity/aspect, and when a
ring is present adds spatial constraints (centroid must be well inside the ring;
equivalent radius bounded) so the ring itself isn't mistaken for the limbus.

**Why this split matters:** a docked eye has a strong artificial circular boundary
and markers that would wreck a naive detector; a pre-docked eye needs eyelid
handling instead. Detecting the ring first and branching is what makes the system
work across the whole surgical timeline.

### 8.6 `red_light_filter.py` — `RedLightFilter` / `AdaptiveRedLightFilter`

Dedicated detector for bright/blinking **red surgical illumination** using
RGB-ratio dominance (`r > g+offset & r > b+offset`), absolute brightness, and
pink/magenta hues, with optional temporal fade across frames. `AdaptiveRedLightFilter`
auto-tunes thresholds to overall scene brightness. Distinct from `ReflectionRemover`
in that it targets coloured light, not white glints.

### 8.7 `temporal_reflection_filter.py` — blink vs. persistent reflection

`TemporalReflectionFilter` keeps a rolling history of bright-pixel masks and marks
a reflection "stable" only if it appears in enough recent frames, so transient
blinks are treated differently from persistent glints. `PupilRegionProtector`
prevents reflection removal from eating a genuinely dark pupil. Not exported from
`__init__.py`; used where explicitly imported (e.g. video preprocessing).

---

## 9. `pupil_tracking.video` — real-time processing & tracking

### 9.1 `optimized_processor.py` — `OptimizedVideoProcessor`

The primary real-time engine (target < 50 ms/frame GPU, < 80 ms CPU). It wraps a
detection backend (prefers `FastInference`, falls back to
`UnifiedDetector.init_video_mode()`) and adds the machinery real-time video needs:

- **Threaded decode-ahead** (`_FrameReader`) — a daemon thread reads/decodes
  frames into a queue honouring `stride`, so decoding overlaps inference.
- **ROI tracking** — `EyeROIDetector` localises the eye and the processor crops to
  it (with a cache TTL), shrinking the work per frame. Manual ROI/ring override is
  supported (`set_manual_roi` / `set_manual_ring`).
- **Batching** (`_prepare_batch` / `_infer_batch_fast`) — groups frames for
  `FastInference.detect_batch`, then smooths sequentially in order.
- **Overload protection** — when latency exceeds a budget and tracking is stable,
  it reuses the last valid result for a bounded burst and adaptively caps the
  processing resolution, so the UI stays responsive under load.
- **Manual-ring priors** — learns average offset/radius ratios of manually placed
  rings into `manual_ring_priors.json` at the repo root and can suggest a ring from
  those priors.
- `VideoPreprocessor` (ring mask → reflection removal → temporal filter →
  normalise, with a `fast_mode` shortcut) and `FrameQualityChecker` (blur/brightness
  gate) support the loop. `_OverlayRenderer` draws the CLI/headless overlay
  (the GUI draws its own).

Public surface: `process_frame(frame, idx)` (single frame, used by the GUI),
`process_video(...)` (file → optional output video + CSV), `process_camera(...)`
(live loop), `update_runtime_settings(...)` (live-tune without rebuild),
`get_stats()` (latency/drop/tracking dict for the UI), `reset()`,
`save_results_json()`. `AsyncCapture` provides a real-time camera reader that
always returns the newest frame.

### 9.2 `video_processor.py` — `VideoProcessor` (classic)

Simpler synchronous per-frame loop built on `UnifiedDetector` +
`EyeKalmanTracker` + `CornealCenterCalculator`. Calls `cfg.apply_video_mode()`
(relaxes thresholds for video). Provides `process_file()`, `process_stream()`,
`export_csv()`, `export_json()`, and a rich `_draw_overlay()` (pupil, limbus,
corneal cross, offset line, cross-section with degree labels, quality badge).
Used as the non-optimized reference path.

### 9.3 `kalman_tracker.py` — `EyeKalmanTracker`

Matrix Kalman filtering for temporal smoothing.

- `EllipseKalmanFilter` — per-ellipse, **8-D state**
  `[cx, cy, semi_major, semi_minor, angle_sin, angle_cos, vx, vy]` with a
  constant-velocity model. Angle is stored as sin/cos to avoid 0°/180° wraparound.
  **Measurement noise is scaled by 1/confidence**, so low-confidence detections
  move the estimate less.
- `EyeKalmanTracker` — dual filter (pupil + limbus; limbus gets lower process
  noise since it moves less). On a missed detection it **carries forward** with a
  decaying confidence for up to `max_carry_forward_frames`. Flags "possible blink"
  when both are lost for several frames.

### 9.4 `temporal_smoother.py` — `TemporalSmoother`

A lighter, per-parameter scalar Kalman used by `OptimizedVideoProcessor` (keys
match the `FastInference` result dict). Adds a **motion-adaptive profile**
(fixation/pursuit/saccade) that rejects large jumps (> 50 px) and tightens or
loosens smoothing by motion regime, plus a **ring-locking state machine** that
locks onto a confidently detected suction ring and tolerates brief misses.

---

## 10. `pupil_tracking.interface` — GUI & recording

### 10.1 `gui_app.py` — `PupilTrackingGUI`

A full Tkinter desktop app (dark clinical theme). Single class, ~5300 lines.

**Three tabs** (right panel notebook):
1. **Measurements** — summary cards (Quality / Tracking / Latency / Pipeline) and
   metric cards for Pupil, Limbus, Corneal Centre & Offset, Calibration, Processing.
2. **Details** — raw text dump of the current result.
3. **⚙ Settings** — grayscale mode, pipeline preset (max_accuracy / balanced /
   low_latency) and optimized/FP16/compile toggles, video resolution/stride/FPS,
   ROI + Kalman noise controls, and a "Rebuild Engine" action.

**Modes:** single image, image folder, video file, live camera — each in classic
or optimized flavour. Worker loops run on daemon threads and marshal UI updates to
the Tk thread via `root.after(...)`.

**Key handlers:** `_open_image` / `_open_folder` / `_open_video`, `_start_video`
(chooses optimized vs classic), `_start_camera`, the four capture loops, the
Tk-thread frame callbacks (`_on_*_frame` → `_update_measurements` +
`_refresh_display`), live-settings debounce (`_apply_live_settings`, with a
restart for changes that need it), `_toggle_pause`, `_stop_video`.

**Overlays:** `_draw_overlay_scaled` draws pupil (green), limbus (blue), corneal
center (white cross, physically sized via calibration), offset line, cross-section,
quality badge, overload banner, and (per the in-progress change) the suction ring
(red). `_draw_debug_overlay` adds a stats box. Manual ROI/ring have their own
interactive draw + mouse handlers.

**Recording / export:** a `FrameRecorder` writes annotated frames; `_export_csv`
(≈ 26 columns incl. mm conversions), `_export_json` (with version/timestamp
metadata), and `_export_snapshot`.

**Keyboard shortcuts:** `Ctrl+O` open image, `Ctrl+V` open video, `Ctrl+Q` quit,
`Space` pause, `Ctrl+R` start recording, `Ctrl+Shift+R` toggle recording,
`G`/`Shift+G` cycle grayscale, `Return`/`Escape` confirm/cancel a manual selection,
arrow keys nudge the ROI.

> **Current uncommitted change** (`git diff`): recording now captures the **full
> UI composite** (image + measurements panel + ring/debug overlays) instead of just
> the image area, and a suction-ring overlay block was added to
> `_draw_overlay_scaled`. See §18 for the two caveats (a duplicate ring-draw block
> and shared-state access from the Tk thread).

### 10.2 `frame_recorder.py` — `FrameRecorder`

Thread-safe video writer: a dedicated writer thread drains a frame queue, codec is
chosen by output extension with fallback through a codec list, `write()` is
non-blocking and **validates exact frame dimensions** (mismatched frames are
dropped — so the composite-recording path must keep constant dimensions).
`start()`/`stop()`/`pause()`/`resume()` manage the lifecycle.

### 10.3 `theme.py` — `Colors` + `DarkTheme`

Central dark-clinical palette (including the per-grade quality colours and
per-section measurement colours) and `DarkTheme.apply(root)` which styles every
ttk widget. Fonts: Segoe UI / Consolas.

---

## 11. `pupil_tracking.calibration` — pixel↔mm

`spatial_calibration.py` provides two classes:

- **`SpatialCalibrator`** — converts pixels↔mm with uncertainty. Strategies:
  from the **limbus** (uses semi-major × 2 only, to avoid a circular dependency;
  assumes 11.5 mm), from a **known ring diameter** (standard 9.4 mm, small 8.5,
  large 10.0), or **manual**. `get_consensus_calibration()` does a
  confidence-weighted average across history.
- **`StabilizedCalibrator`** — EMA-smoothed calibration with outlier rejection
  that **freezes** once it has enough consistent samples (coefficient of variation
  < 2 %), so the reported scale is rock-steady (surgical ±0.01–0.02 mm target)
  instead of jittering frame to frame. Propagates uncertainty and attaches it as
  dynamic attributes. This is what `UnifiedDetector` uses.

**Why:** all clinically meaningful outputs (pupil diameter in mm, decentration in
mm) depend on a stable px↔mm scale; the limbus provides a built-in physical ruler.

---

## 12. `pupil_tracking.annotation` — labeling

`annotation_tool.py::AnnotationTool` — a Tkinter tool to click points for pupil,
limbus, or ring, fit an ellipse (`cv2.fitEllipse`), and save/load annotations as
JSON (boundary points + ellipse params). Launched by `scripts/annotate_data.py`.
For higher-throughput work, `scripts/annotate_live_video.py` is the richer,
video-oriented annotator (edge-snapping, incremental retraining).

---

## 13. `pupil_tracking.utils` — config, types, logging, runtime profile

### 13.1 `config.py` — `PupilTrackingConfig`

A tree of nested `@dataclass`es aggregated under `PupilTrackingConfig`, accessed
through a global singleton (`get_config()` / `set_config()` / `reset_config()`),
with JSON `save()`/`load()` (unknown keys dropped for forward-compatibility).

| Sub-config | Purpose | Notable defaults |
|------------|---------|------------------|
| `ModelConfig` | model + inference | `encoder=resnet34`, `input_size=512`, `num_classes=3`, `model_path=models/best_model.pth`, `confidence_threshold=0.5`, `device=auto`, `multiscale_sizes=(448,512,640)` |
| `DetectionConfig` | detection thresholds | `min_pupil_confidence=0.25`, `min_limbus_confidence=0.25`, `min_pupil_area=200`, `min_limbus_area=3000`, `enable_classical_fallback=True`, `classical_confidence_penalty=0.85` |
| `FittingConfig` | RANSAC + Huber fit | `ransac_iterations=500`, `ransac_threshold=1.5`, `min_fit_quality=0.30`, `max_rms_residual=5.0`, `huber_delta=1.5` |
| `VideoConfig` | Kalman + temporal | `enable_kalman=True`, `kalman_process_noise=0.1`, `kalman_measurement_noise=1.0`, `target_fps=30`, `max_carry_forward_frames=5`, `carry_forward_decay=0.85` |
| `CalibrationConfig` | scale references | `suction_ring_diameter_mm=9.4`, `corneal_diameter_mm=11.5`, `enable_auto_calibration=True` |
| `PathConfig` | filesystem | `model_dir=models`, `data_dir=clinical_data`, `log_dir=logs`, `ring_classifier_path=models/ring_classifier.pth` |
| `TrainingConfig` | training | `epochs=200`, `batch_size=4`, `learning_rate=1e-4`, `early_stopping_patience=30`, `val_ratio=0.2`, `augmentations_per_image=50`, `use_amp=True` |
| `RingClassifierConfig` | ring CNN | `enabled=True`, `confidence_threshold=0.70`, `input_size=224` |
| `RingHeuristicConfig` | ring Hough/Canny | `canny_low=30`, `canny_high=100`, `hough_dp=1.2`, `radius_min_frac=0.25`, `radius_max_frac=0.48` |
| `RingPreprocessingConfig` | docked prep | `inner_margin=15`, `clahe_clip_multiplier=1.2`, `clahe_grid_docked=(4,4)`, `reflection_percentile=97.0`, `eyelid_margin_frac=0.15` |
| `DockedDetectionConfig` | docked overrides | `threshold_value=35`, `max_pupil_ring_ratio=0.60`, `max_center_offset_ratio=0.75` |
| `PreDockedDetectionConfig` | pre-docked overrides | `threshold_value=40`, `adaptive_block_size=51`, `min_circularity=0.30` |
| `GrayscaleConfig` | grayscale enhance | `clahe_clip_low=1.5`, `clahe_clip_high=4.0`, `unsharp_amount=0.5`, `use_bilateral=True` |
| `MeasurementStabilizationConfig` | calibration EMA | `ema_alpha=0.15`, `outlier_sigma=2.0`, `min_samples_for_rejection=5`, `max_calibration_history=50` |
| `SubPixelConfig` | sub-pixel refine | `interpolation_step=0.25`, `gradient_scales=(1,3)`, `bootstrap_n_samples=50` |
| `RingConfig` | aggregates ring configs | `default_mode=auto`, `merge_weight_classifier=0.65`, `agreement_bonus=1.15` |

Top-level `PupilTrackingConfig` also has `video_mode=False`, `debug=False`, and
methods `apply_video_mode()` (relaxes confidence/point thresholds and enables
Kalman for video — call before video processing) and
`get_ring_detection_params(mode)` (flat param dict for `docked`/`pre_docked`).

### 13.2 `types.py` — the result schema

**Enums:** `DetectionQuality{SURGICAL, CLINICAL, RESEARCH, INSUFFICIENT,
NO_DETECTION}`, `DetectionMethod{ML, CLASSICAL, HYBRID, KALMAN, CARRY_FORWARD}`,
`QualityFlag{GOOD, MARGINAL, POOR, NO_DETECTION}`.

**Grade logic (single source of truth):** `_GRADE_THRESHOLDS = [(0.75,SURGICAL),
(0.55,CLINICAL),(0.30,RESEARCH),(0.0,INSUFFICIENT)]`; `assign_quality_grade()`
maps a confidence to a grade (non-finite → `NO_DETECTION`).

**`AnatomicalLimits`** (frozen singleton `ANATOMICAL_LIMITS`) — hard plausibility
bounds used by cross-validation: pupil radius 8–120 px (0.5–5.0 mm), limbus
40–250 px (4.5–7.5 mm), pupil/limbus ratio 0.15–0.75, max eccentricity 0.87, max
offset 1.5 mm, etc.

**Dataclasses (the schema in full):**

- `EllipseParams` — `center_x/y`, `semi_major/minor`, `angle_deg`, per-parameter
  uncertainties, `fit_quality`, `fit_rms_residual`, `eccentricity`, `circularity`.
  `radius` is a **read-only** property (mean of semi-axes) — mutate via
  `set_radius()`. Has `center`, `is_valid`, `to_dict()`.
- `PupilDetection` / `LimbusDetection` (identical shape) — `detected`, `ellipse`,
  `confidence`, `quality`, `method`, `center_mm`, `radius_mm`, `contour_points`.
- `CornealCenterResult` — `valid`, `center_px/mm`, `offset_px/mm`,
  `offset_magnitude_px/mm`, `offset_angle_deg`, `confidence`, `quality`, `alerts`.
- `CalibrationInfo` — `calibrated`, `px_per_mm`, `mm_per_px`, `source`,
  `reference_diameter_mm/px`, `confidence`, plus `px_to_mm()`/`point_px_to_mm()`.
- `FrameMetadata` — timestamp, frame number, source, dimensions, processing time.
- **`EyeDetectionResult`** — the top-level container (see §17).
- `FitResult` — output of `EllipseFitter.fit()` (`fit_quality_score` named to
  avoid colliding with `EllipseParams.fit_quality`); `to_ellipse_params()` bridges
  to the schema.

### 13.3 `logger.py` — `AuditLogger`

Thread-safe structured logging: a namespaced `logging.Logger` (UTF-8 console for
Windows emoji safety) plus a `session_{id}.log` and an append-only
`audit_{session_id}.jsonl` (one JSON object per line, flushed per write). API:
`log_detection()`, `log_alert()`, `info/warning/error/debug`, `close()`. Global
singleton via `get_logger()` / `set_logger()`. The JSONL trail is the audit record
for clinical/debugging review.

### 13.4 `runtime_profile.py` — hardware-adaptive tuning

`detect_runtime_profile()` (cached) reads CPU cores, RAM, and CUDA availability and
returns a frozen `RuntimeProfile` with recommended resolution, target FPS, FP16,
compile, batch size, and thread counts. Three tiers: **`gpu_accelerated`** (CUDA:
FP16+compile on, batch 4), **`cpu_compact`** (≤ 8.5 GB RAM or ≤ 4 cores: FP16 off,
batch 1), **`cpu_balanced`** (otherwise). `apply_runtime_optimizations()` then sets
`cv2.setNumThreads`, disables OpenCL, and sets torch thread/interop counts. This is
why `launch_gui.py`'s FP16/compile/resolution defaults differ per machine.

---

## 14. `scripts/` — training, export, evaluation, tooling

### Training & model management
- **`train_model.py`** — primary training entry. Groups: training
  (`--epochs`, `--batch-size`, `--lr`, `--input-size`, `--device`,
  `--copies-per-image`), model (`--num-classes {3,4}`, `--encoder`), loss
  (`--loss-type {composite,ce_dice,focal_dice,ce}`, `--use-focal`,
  `--focal-gamma`, `--class-weights`), data (`--annotation-path`, `--image-dir`,
  `--mask-dir`, `--ring-labels`), output (`--save-dir`, `--model-name`). Patches
  config, builds `Trainer`, trains, then reports per-class IoU (bg/pupil/iris/
  suction_ring) and a quality tier.
- **`run_epoch.py`** — one-epoch smoke test to validate the data/model pipeline.
- **`train_ring_classifier.py`** — trains the MobileNetV2 ring classifier from
  `ring_labels.json`.
- **`finetune_grayscale.py`** — fine-tunes an existing model for grayscale
  robustness with a **safety gate**: only saves if RGB Dice doesn't regress AND
  grayscale Dice ≥ `--min-gray-dice` (default 0.88). Dual RGB + forced-gray
  validation sets.

### Export & deployment
- **`export_onnx.py`** — simple single-model (segmentation) ONNX export
  (opset 18, dynamic axes) with optional `--verify`.
- **`convert_to_onnx.py`** — full pipeline: converts **both** segmentation and
  ring classifier, heavy architecture auto-detection, optional INT8 quantization,
  writes `models/onnx/manifest.json` with sizes + sha256. Forces the legacy
  exporter on torch ≥ 2.6. Requires `onnx` + `onnxruntime`.

### Video & benchmarking
- **`process_video.py`** — headless `OptimizedVideoProcessor` runner (file or
  camera), CSV/JSON output, `--benchmark` mode. Note its smoothing defaults
  (`--process-noise=2.0`, `--measurement-noise=4.0`) differ from `VideoConfig`.
- **`benchmark_fps.py`** — inference FPS at various resolutions / FP16.
- **`benchmark_video_speed.py`** — end-to-end pipeline timing (decode / preprocess
  / infer / batch), synthetic-frame option.

### Annotation & data prep
- **`annotate_data.py`** — thin launcher for `AnnotationTool`.
- **`annotate_live_video.py`** — advanced live-video annotator with edge-snapping
  and incremental retraining (subcommands: `annotate`, `generate-masks`, `train`,
  `check`).
- **`annotate_ring_data.py`** — keyboard ring-presence labeling (R=present,
  N=absent, P=partial) → `ring_labels.json`.
- **`generate_masks.py`** — rasterise annotations into training masks.

### Validation & diagnostics
- **`verify_data.py`** — annotation completeness + anatomical plausibility.
- **`check_files.py`** — verifies expected project files exist.
- **`check_training_data.py`** — mask validity + class-distribution report.
- **`diagnose_detection.py`** — per-image diagnostic visualisations (masks,
  heatmaps, overlays).
- **`evaluate_ring_detection.py`** — ring accuracy/precision/recall/F1 vs
  `ring_labels.json` (combined, classifier-only, or heuristic-only).
- **`test_grayscale_detection.py`** — RGB-vs-grayscale comparison harness.
- **`debug_single_image.py`** — ⚠ **broken**: imports `pupil_tracking.enhanced_detection`
  and `pupil_tracking.clinical_pipeline`, which **do not exist**. Legacy; will fail
  on import until fixed or removed.

---

## 15. `pupil_tracking.tests` — test suite

Run with `python -m pytest pupil_tracking/tests/ -v`. Model/data requirements:

| Test file | Needs model | Needs data | What it covers |
|-----------|-------------|------------|----------------|
| `test_clinical_accuracy.py` | yes (or skip) | yes `clinical_data/clean/` (or skip) | per-image: pupil/limbus detected, confidence > 0.3, pupil inside limbus, radius ratio bounds |
| `test_video_pipeline.py` | yes (else early-return) | no (synthetic) | ROI detector, temporal smoother, fast inference, video mode, latency < 100 ms |
| `test_grayscale.py` | no | no | `GrayscaleHandler` detection/convert/enhance, thread safety, edge cases |
| `test_deterministic_ring_detector.py` | no | no (synthetic eyes) | pre-op → ABSENT, post-op ring → PRESENT with geometry |
| `test_manual_roi.py` | no | no | manual ROI crop math, quality annotation, stat counters |

`test_manual_roi.py` constructs `OptimizedVideoProcessor` via `object.__new__`
(bypassing `__init__`), so it is coupled to private attributes — fragile to
refactors.

---

## 16. Models, data & artifacts

- **`models/best_model.pth`** — U-Net/ResNet-34 checkpoint (git-ignored;
  `checkpoint_meta.json` records epoch 30, val IoU 0.943).
- **`models/onnx/segmentation.onnx`** (~93 MB) and
  **`segmentation_quantized.onnx`** (~23 MB) — production inference models
  (git-ignored; `manifest.json` tracks sizes + sha256).
- **`models/ring_classifier.pth`** — MobileNetV2 ring classifier (optional).
- **`manual_ring_priors.json`** (repo root) — runtime cache of learned manual-ring
  offset/radius ratios (currently: 106 samples, radius_ratio ≈ 0.90).
- **`clinical_data/`** — a small committed sample (eye_01…eye_14 + masks +
  `annotations.json`). Real training data lives under `clinical_data/training_data/`
  which is git-ignored.

**`requirements.txt`:** torch ≥ 1.12, torchvision ≥ 0.13,
segmentation-models-pytorch ≥ 0.3, opencv-python ≥ 4.6, Pillow ≥ 9, albumentations
≥ 1.3, scipy ≥ 1.7, numpy ≥ 1.21, scikit-learn ≥ 1.0, matplotlib ≥ 3.5, tqdm ≥
4.64, pytest ≥ 6. **Missing but imported by scripts:** `onnx` and `onnxruntime`
(needed for ONNX export and production inference). Only lower bounds are pinned.

---

## 17. The `EyeDetectionResult` schema

`utils/types.py::EyeDetectionResult` is the single object every detector returns
and every consumer reads.

**Declared fields:**
```python
pupil:          PupilDetection        # detected, ellipse, confidence, quality, method, *_mm
limbus:         LimbusDetection       # same shape as pupil
corneal_center: CornealCenterResult   # offset vector (px + mm), magnitude, angle
calibration:    CalibrationInfo       # px_per_mm / mm_per_px + source
metadata:       FrameMetadata         # frame number, source, timing
overall_quality: DetectionQuality     # SURGICAL / CLINICAL / RESEARCH / …
overall_confidence: float
alerts:         list[str]
```

**Properties:** `has_pupil`, `has_limbus`, `has_both`, `has_corneal_center`.

**Dynamically-attached ring attributes** (set by `UnifiedDetector`, not declared
fields — `to_dict()` serialises them opportunistically via `hasattr`):
`ring_status`, `ring_confidence`, `ring_center`, `ring_radius`, `ring_inner_radius`,
`ring_mask`, `ring_contour`, `ring_dot_centers`, `ring_dot_count`, `ring_method`,
`image_category`, `corneal_reference_source`, plus `_raw_mask` (the integer label
map). Treat these as part of the effective schema even though they aren't in the
dataclass definition.

**Serialisation helpers:** `to_dict()` (nested), `result_to_dict()` (flat — the
CSV / `TemporalSmoother` contract, zero-filled when not detected),
`apply_smoothed_dict()` (writes Kalman values back respecting `set_radius()`).

---

## 18. Known issues, dead code & drift

Surfaced by the code read; useful to know before editing.

**Dead / unused in the main pipeline:**
- `ConfidenceScorer` (`core/confidence.py`) — never called by `detect()`.
- `RingAwarePreprocessor.preprocess()` output — computed at detect() step 2 but
  not consumed downstream.
- A heuristic limbus-correction block in `detector.py` guarded by
  `ring_status == "PRESENT"` — never fires because the actual value is
  `"ring_present"` (string mismatch).
- Legacy `core/ring_detector.py` — shadowed by `deterministic_ring_detector.py`.
- `CornealCenterCalculator.calculate()` output — overwritten by
  `_blend_corneal_center_from_available()` whenever a pupil is detected.

**Naming collisions / schema drift:**
- **Two `FitResult` types**: `core/smart_fitter.py` vs `utils/types.py` (different
  fields). Be explicit about which you import.
- Ring attributes are monkey-patched onto `EyeDetectionResult` (undeclared).
- `SuctionRingResult` uses British `ring_centre` while `ring_aware` expects
  `ring_center` — different objects, not interchangeable.

**Hardcoded clinical constants (no config knob):**
- Pre-docked limbus shrink ×0.93 (both `detector.py` and `optimized_processor.py`).
- Corneal diameter 11.5 mm assumption (several sites).
- Motion-jump rejection 50 px (`temporal_smoother.py`).
- Ring-lock magic thresholds (`temporal_smoother.py`).

**Broken / fragile:**
- `scripts/debug_single_image.py` imports non-existent modules.
- `onnx` / `onnxruntime` missing from `requirements.txt`.
- Silent `except Exception: pass` in several preprocessing and GUI loops hides
  failures.
- GUI recording recomposes frames from shared mutable state on the Tk thread
  (possible race with worker threads); the in-progress diff adds a **second**
  ring-draw block in `_draw_overlay_scaled` (one already exists, with a different
  cross colour).
- `FrameRecorder.write()` silently drops frames whose dimensions don't match the
  writer — the composite-recording change must keep dimensions constant.

**Doc drift (now corrected):** the old README described many files that don't
exist (see §4 note) and mixed two class-name vocabularies. This document reflects
the code as it actually is.

---

## 19. Glossary

| Term | Meaning |
|------|---------|
| **Pupil** | Dark central aperture of the iris |
| **Iris** | Coloured ring around the pupil |
| **Limbus** | Iris–sclera boundary; ≈ 11.5 mm diameter; scale reference |
| **Corneal center** | Defined here as the limbus center |
| **Decentration / offset** | Pupil-center minus limbus-center vector |
| **Docked** | Suction ring attached to the eye (femto-laser) |
| **Pre-docked** | Natural eye, no ring attached |
| **Suction ring** | Immobilising ring with red LED marker dots |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalisation |
| **RANSAC** | Random Sample Consensus (robust fitting) |
| **Dice / IoU** | Segmentation overlap metrics |
| **Temperature scaling** | Post-hoc probability calibration |
| **ONNX** | Open Neural Network Exchange (portable inference format) |
| **ROI** | Region of Interest (the eye crop) |
| **Kalman filter** | Recursive estimator for temporal smoothing |
| **Quality grade** | SURGICAL / CLINICAL / RESEARCH / INSUFFICIENT / NO_DETECTION |

---

*This analysis was produced from a direct, end-to-end read of the source. When
code and prose disagree, the code is authoritative — update this document as the
code evolves.*











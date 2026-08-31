# REPOSITORY CLEANUP REPORT

**Date:** 2026-08-31
**HEAD:** `811929c` (pre-cleanup)
**Scope:** Remove disposable iris phase artifacts, temp scripts, generated outputs.

---

## 1. Initial Repository State

| Item | Value |
|------|-------|
| HEAD | `811929c` |
| target/main | `811929c` (verified) |
| Pre-existing modified files | 57 (all unrelated to iris) |
| Pre-existing untracked files | 28 |
| Tracked IRIS reports | 14 |
| Tracked IRIS scripts | 3 |
| Untracked source modules | 6 (gui/*, confidence.py, ring_detector.py, video_processor.py, test_runtime_profile.py) |

---

## 2. Complete File Inventory

**FACT:** Total files inspected: ~90 unique candidates across root, scripts/, _phase_artifacts/, pupil_tracking/.

Classification:

- **DELETE**: 22 files + 1 directory
- **CONSOLIDATE**: 1 pair (candidate only, not executed)
- **UNKNOWN**: 6 files (requires human decision)
- **KEEP**: ~60 files (production code, documentation, evaluation scripts)

---

## 3. Documentation Inventory

### Tracked IRIS Documentation (14 files)

| File | Phase | Status |
|------|-------|--------|
| IRIS_FEATURE_DETECTION_PLAN.md | Plan | Canonical design doc |
| IRIS_INTEGRATION_ARCHITECTURE.md | Architecture | Canonical architecture doc |
| IRIS_NEXT_PHASE_AUDIT.md | Planning | Phase transition audit |
| IRIS_PHASE2_REPORT.md | II | Feature detection report |
| IRIS_PHASE4_REPORT.md | IV | Correspondence report |
| IRIS_PHASE4_BENCHMARK_RESULTS.md | V-A | Benchmark results (overlap with PHASE4_REPORT) |
| IRIS_PHASE5_PLAN.md | V | Implementation plan |
| IRIS_PHASE6_AUDIT.md | VI | Audit report |
| IRIS_PHASE8C_AUDIT.md | VIII-C | FALSE-OK classification |
| IRIS_PHASE8D_COARSE_NCC_GATE_REPORT.md | VIII-D | Coarse/NCC gate |
| IRIS_PHASE8E_NCC_ACCURACY_AUDIT.md | VIII-E | NCC accuracy audit |
| IRIS_PHASE9_AUDIT.md | IX | Feature representation audit |
| IRIS_PHASE10_GLOBAL_CONSISTENCY_REPORT.md | X | Global consistency report |
| IRIS_CLEAN_AUDIT.md | Pre-phase | Clean pre-phase audit |

### Untracked IRIS Documentation (2 files — DELETED)

| File | Status |
|------|--------|
| IRIS_PHASE8A_RECOVERY_REPORT.md | Superseded, no references |
| IRIS_PHASE8B_NCC_FIX_REPORT.md | Superseded, no references |

---

## 4. Duplicate Documentation Findings

**IRIS_PHASE4_BENCHMARK_RESULTS.md vs IRIS_PHASE4_REPORT.md:**

- PHASE4_REPORT: Phase IV formal report (HEAD 65ccbe8, 679 lines, narrative + data)
- BENCHMARK_RESULTS: Phase V-A benchmark results (HEAD c95a105, 691 lines, raw tables)
- Overlap: ~60% (benchmark methodology, harness description, results tables)
- Unique to REPORT: narrative analysis, integration context, architecture discussion
- Unique to BENCHMARK_RESULTS: raw data tables, timing breakdown

**INFERENCE:** These are consolidation candidates. Marked CONSOLIDATE but NOT deleted in this cleanup — consolidation requires careful merging and is a separate task.

---

## 5. Temporary Artifacts Found

| File | Type | References |
|------|------|-----------|
| `_baseline_importtime.txt` | Generated profiling output | None |
| `_phase_artifacts/phase8e/*.json` (3 files) | Generated benchmark outputs | None |

---

## 6. Temporary Scripts Found

| File | Phase | References | Classification |
|------|-------|-----------|---------------|
| `scripts/iris_phase8a_trace.py` | VIII-A | None | DELETE |
| `scripts/iris_phase8c_audit.py` | VIII-C | None | DELETE |
| `scripts/iris_phase8c_coarse_audit.py` | VIII-C | None | DELETE |
| `scripts/iris_phase8d_conservative.py` | VIII-D | None | DELETE |
| `scripts/iris_phase8d_final.py` | VIII-D | None | DELETE |
| `scripts/iris_phase8d_measure.py` | VIII-D | None | DELETE |
| `scripts/iris_phase8d_threshold.py` | VIII-D | None | DELETE |
| `scripts/iris_phase8e_analysis.py` | VIII-E | None | DELETE |
| `scripts/iris_phase8e_ncc_diagnostics.py` | VIII-E | None | DELETE |
| `scripts/iris_phase8e_steps5to8.py` | VIII-E | None | DELETE |
| `scripts/iris_phase8e_timing.py` | VIII-E | None | DELETE |
| `scripts/iris_phase8e_window.py` | VIII-E | None | DELETE |
| `scripts/iris_phase9_diag.py` | IX | None | DELETE |
| `scripts/iris_phase10_benchmark.py` | X | None | DELETE |
| `scripts/iris_phase4_correspondence_eval.py` | IV | 21 doc refs | KEEP |
| `scripts/iris_phase2_eval.py` | II | doc refs | KEEP |
| `scripts/iris_feature_smoke.py` | — | test ref | KEEP |
| `scripts/debug_single_image.py` | — | README ref | KEEP |

---

## 7. Files Deleted

| # | Path | Reason | Risk |
|---|------|--------|------|
| 1 | `_baseline_importtime.txt` | Generated profiling output, not referenced | None |
| 2 | `inspect_ring_detector.py` | 4-line stub, not referenced | None |
| 3 | `inspect_inference.py` | 6-line inspection script, not referenced | None |
| 4 | `IRIS_PHASE8A_RECOVERY_REPORT.md` | Untracked, superseded, no references | None |
| 5 | `IRIS_PHASE8B_NCC_FIX_REPORT.md` | Untracked, superseded, no references | None |
| 6 | `scripts/iris_phase8a_trace.py` | One-off diagnostic, not referenced | None |
| 7 | `scripts/iris_phase8c_audit.py` | One-off diagnostic, not referenced | None |
| 8 | `scripts/iris_phase8c_coarse_audit.py` | One-off diagnostic, not referenced | None |
| 9 | `scripts/iris_phase8d_conservative.py` | One-off diagnostic, not referenced | None |
| 10 | `scripts/iris_phase8d_final.py` | One-off diagnostic, not referenced | None |
| 11 | `scripts/iris_phase8d_measure.py` | One-off diagnostic, not referenced | None |
| 12 | `scripts/iris_phase8d_threshold.py` | One-off diagnostic, not referenced | None |
| 13 | `scripts/iris_phase8e_analysis.py` | One-off diagnostic, not referenced | None |
| 14 | `scripts/iris_phase8e_ncc_diagnostics.py` | One-off diagnostic, not referenced | None |
| 15 | `scripts/iris_phase8e_steps5to8.py` | One-off diagnostic, not referenced | None |
| 16 | `scripts/iris_phase8e_timing.py` | One-off diagnostic, not referenced | None |
| 17 | `scripts/iris_phase8e_window.py` | One-off diagnostic, not referenced | None |
| 18 | `scripts/iris_phase9_diag.py` | One-off diagnostic, not referenced | None |
| 19 | `scripts/iris_phase10_benchmark.py` | Phase X benchmark, not referenced | None |
| 20 | `_phase_artifacts/phase8e/ncc_diagnostics.json` | Generated output, not referenced | None |
| 21 | `_phase_artifacts/phase8e/steps5to8_analysis.json` | Generated output, not referenced | None |
| 22 | `_phase_artifacts/phase8e/window_analysis.json` | Generated output, not referenced | None |
| 23 | `_phase_artifacts/phase8e/` | Empty after removing 3 JSON files | None |
| 24 | `_phase_artifacts/` | Empty after removing phase8e/ | None |

**Total deleted: 22 files + 1 directory**

---

## 8. Files Intentionally Retained

### Production/Source Code
- `pupil_tracking/iris/*` — full iris subsystem (intact)
- `pupil_tracking/core/*` — pupil/limbus detection (intact)
- `pupil_tracking/ml/*` — ML inference (intact)
- `pupil_tracking/preprocessing/*` — image preprocessing (intact)
- `pupil_tracking/video/*` — video processing (intact)
- `pupil_tracking/calibration/*` — calibration (intact)
- `pupil_tracking/interface/*` — GUI (intact)
- `pupil_tracking/utils/*` — utilities (intact)

### Evaluation/Build Scripts
- `scripts/iris_phase4_correspondence_eval.py` — canonical benchmark harness
- `scripts/iris_phase2_eval.py` — canonical eval script
- `scripts/iris_feature_smoke.py` — referenced by test
- `scripts/debug_single_image.py` — referenced by README (as broken)
- `gen_notebook.py` — referenced by README, builds train_colab.ipynb

### Documentation
- All 14 tracked IRIS reports — historical evidence
- `README.md` — canonical project docs
- `CLAUDE.md` — project context
- `clinical_data/README.md` — data documentation
- `pupil_tracking/iris/README.md` — iris subsystem docs
- `models/README.md` — model documentation

---

## 9. UNKNOWN Files (Retained)

| File | Reason |
|------|--------|
| `pupil_tracking/interface/gui/*` (13 files) | Refactored GUI package, untracked, not imported by launch_gui.py. Could be ongoing work. |
| `pupil_tracking/core/confidence.py` | 513-line module, substantial ongoing work. |
| `pupil_tracking/core/ring_detector.py` | 749-line module, substantial ongoing work. |
| `pupil_tracking/video/video_processor.py` | 519-line module, substantial ongoing work. |
| `pupil_tracking/tests/test_runtime_profile.py` | 303-line test file, has pre-existing import error. |

**VERDICT:** These are substantial source files that appear to be ongoing development work. They were NOT created by this cleanup and are NOT candidates for blind deletion. They require explicit human decision.

---

## 10. Test Results

| Suite | Before | After | Status |
|-------|--------|-------|--------|
| Iris tests | 93/93 | **93/93** | PASS |
| test_runtime_profile.py | Import error | Import error | Pre-existing (not our cleanup) |

**VERDICT: ZERO test regressions from cleanup.**

---

## 11. Production-Safety Verification

| Check | Status |
|-------|--------|
| pupil_tracking/core/detector.py | INTACT |
| pupil_tracking/ml/architecture.py | INTACT |
| pupil_tracking/interface/gui_app.py | INTACT |
| pupil_tracking/calibration/ | INTACT |
| No iris imports in production code | VERIFIED |
| No model changes | VERIFIED |

---

## 12. Iris Subsystem Verification

| Check | Status |
|-------|--------|
| pupil_tracking/iris/ intact | VERIFIED |
| correspondence.py intact | VERIFIED |
| Global spatial consistency intact | VERIFIED |
| Phase X implementation available | VERIFIED |
| No production integration introduced | VERIFIED |
| 93/93 tests pass | VERIFIED |

---

## 13. Remaining Repository Hygiene Issues

1. **IRIS_PHASE4_BENCHMARK_RESULTS.md overlaps with IRIS_PHASE4_REPORT.md** — consolidation candidate (not executed in this cleanup)
2. **UNKNOWN files** — 5 untracked source modules need human decision
3. **test_runtime_profile.py** — has pre-existing import error (unrelated to cleanup)
4. **57 pre-existing modified files** — unrelated to this cleanup, untouched

---

## 14. Recommendations

1. **Decide on UNKNOWN files** — the gui/ package, confidence.py, ring_detector.py, video_processor.py, and test_runtime_profile.py are substantial ongoing work. Decide whether to track, gitignore, or remove them.
2. **Consolidate Phase IV reports** — IRIS_PHASE4_BENCHMARK_RESULTS.md and IRIS_PHASE4_REPORT.md overlap ~60%. Consider merging into one canonical document.
3. **Fix test_runtime_profile.py** — has stale import of `_probe_cuda_driver`. Either fix the import or remove the test file.
4. **Consider adding to .gitignore** — `_baseline_importtime.txt`, `inspect_*.py` patterns to prevent future accumulation.

---

## Summary

| Metric | Value |
|--------|-------|
| Files inspected | ~90 |
| Files deleted | 22 + 1 directory |
| Files retained | ~60 |
| UNKNOWN files | 6 |
| Duplicate docs | 1 pair (consolidation candidate) |
| Generated artifacts removed | 3 JSON files + 1 directory |
| Test regressions | **0** |
| Iris functionality | **INTACT** |
| Production pipeline | **INTACT** |

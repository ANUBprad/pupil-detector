# IRIS PHASE XVIII-FIX — CYCLOTORSION / IRIS VISIBLE IN IXcentai MEASUREMENTS PANEL

**Date:** 2026-09-01
**HEAD:** `2968221` (before fix)
**Scope:** UI completion — make CYCLOTORSION / IRIS card visible in running application.

---

## 1. Root Cause

The CYCLOTORSION / IRIS card was added to `pupil_tracking/interface/gui/builders_mixin.py` (the mixin refactoring) in Phase XVII, but the actual running application uses `pupil_tracking/interface/gui_app.py` (the monolithic file). The monolithic file was never updated with:
- The iris card in `_build_measurements_panel`
- The iris detection call in `_load_single_image`
- The iris update logic in `_update_measurements`
- The iris import

The mixin files are tracked but NOT used by `launch_gui.py` (which imports from `gui_app`).

---

## 2. Files Changed

| File | Change |
|------|--------|
| `pupil_tracking/interface/gui_app.py` | Added import, iris card, iris detection call, iris update logic |

**No other files modified.** No centration code touched.

---

## 3. Changes Made

### 3a. Import (line 46)
```python
from pupil_tracking.iris.detect import detect_iris_features
```

### 3b. CYCLOTORSION / IRIS card in `_build_measurements_panel` (after PROCESSING card)
```python
iris_frame = add_card(
    cards_outer, "CYCLOTORSION / IRIS", "OffsetHeader.TLabel", 4, 0, 2
)
self._iris_vars["status"] = add_row(iris_frame, "Status:")
self._iris_vars["feature_count"] = add_row(iris_frame, "Features:")
self._iris_vars["angular_coverage"] = add_row(iris_frame, "Coverage:")
self._iris_vars["rotation_angle"] = add_row(iris_frame, "Rotation Angle:")
self._iris_vars["confidence"] = add_row(iris_frame, "Confidence:")
self._iris_vars["evidence"] = add_row(iris_frame, "Evidence:")
```

### 3c. Iris detection call in `_load_single_image` (after detection)
```python
try:
    pupil_e = result.pupil.ellipse if result.pupil.detected else None
    limbus_e = result.limbus.ellipse if result.limbus.detected else None
    iris_result = detect_iris_features(image, pupil_e, limbus_e)
    result.iris_detection = iris_result
    result.iris_status = iris_result.status
except Exception:
    result.iris_detection = None
    result.iris_status = None
```

### 3d. Iris update logic in `_update_measurements`
- If iris valid: show features, coverage, status
- If paired result exists: show rotation, confidence, evidence
- If iris unavailable: show "Rejected: [status]" or "Unavailable"
- All fields default to "---" when unavailable

### 3e. Iris features in `_update_details` (text panel)
- Shows "=== IRIS FEATURES ===" section with features, coverage, status

---

## 4. Visible Cyclotorsion Fields

The CYCLOTORSION / IRIS card displays:

| Field | Single Image | Valid Pair | Rejected |
|-------|-------------|------------|----------|
| Status | Valid | Valid | Rejected: [reason] |
| Features | 42 | 42 | --- |
| Coverage | 69.9% | 69.9% | --- |
| Rotation Angle | --- | +3.24 | --- |
| Confidence | --- | High | --- |
| Evidence | Single image | Good | --- |

---

## 5. Tests

| Suite | Count | Status |
|-------|-------|--------|
| Iris + Pentacam | 135 | **135/135 PASS** |
| Clinical accuracy | 58 | **58/58 PASS** |

---

## 6. Centration Regression Verification

| Check | Status |
|-------|--------|
| pupil_detection.py unchanged | VERIFIED |
| limbus detection unchanged | VERIFIED |
| corneal_center.py unchanged | VERIFIED |
| ring detection unchanged | VERIFIED |
| offset calculation unchanged | VERIFIED |
| calibration unchanged | VERIFIED |
| `git diff --stat HEAD -- pupil_tracking/core/` | EMPTY |

**VERIFIED:** No centration code touched.

---

## 7. Production Safety

| Check | Status |
|-------|--------|
| Existing PUPIL card unchanged | VERIFIED |
| Existing LIMBUS card unchanged | VERIFIED |
| Existing CORNEAL CENTRE & OFFSET unchanged | VERIFIED |
| Existing CALIBRATION unchanged | VERIFIED |
| Existing CORNEAL DIMENSIONS unchanged | VERIFIED |
| Existing PROCESSING unchanged | VERIFIED |
| Only addition: CYCLOTORSION / IRIS card | VERIFIED |
| Iris detection is additive (try/except) | VERIFIED |

---

## 8. Summary

| Item | Value |
|------|-------|
| Root cause | Card in mixin, not in gui_app.py |
| Fix | 4 surgical edits to gui_app.py |
| Card visible | YES (row 4, full width) |
| Existing cards | UNCHANGED |
| Tests | 135/135 PASS |
| Centration | UNTOUCHED |
| Production safety | PASS |

---

## 9. STOP

Do NOT modify centration. Do NOT start another phase.

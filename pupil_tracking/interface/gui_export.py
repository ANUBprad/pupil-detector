# pupil_tracking/interface/gui_export.py
"""Export functions (CSV, JSON, snapshot) extracted from
:class:`PupilTrackingGUI` during the Phase-4 refactoring.

These are free functions that accept explicit parameters instead
of relying on instance state, making them independently testable.
"""

from __future__ import annotations

import csv
import json
import math
import time
from typing import Any, Dict, List, Optional

import cv2

# ── Constants ──────────────────────────────────────────────────────

from pupil_tracking.utils.config import CORNEAL_DIAMETER_MM as _CORNEAL_DIAMETER_MM

# ── CSV export ─────────────────────────────────────────────────────


def export_csv(
    results_history: List[Dict[str, Any]],
    status_var: Any,
) -> Optional[str]:
    """Export results history to CSV. Returns the path on success, None on cancel.

    Parameters
    ----------
    results_history : list
        List of result dicts to export.
    status_var : tk.StringVar
        Status bar variable to update on success.

    Returns
    -------
    str or None
        The export path, or None if cancelled/no data.
    """
    from tkinter import filedialog, messagebox

    if not results_history:
        messagebox.showinfo("No Data", "No results to export")
        return None

    path = filedialog.asksaveasfilename(
        title="Export CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
    )
    if not path:
        return None

    def _round(v: Any, nd: int = 4) -> Any:
        if v is None or v == "":
            return ""
        try:
            fv = float(v)
            if not math.isfinite(fv):
                return ""
            return round(fv, nd)
        except (TypeError, ValueError):
            return v

    def _diam_mm(ell: Dict[str, Any], mm_px: float) -> Any:
        if ell.get("diameter_mm") not in (None, ""):
            return _round(ell.get("diameter_mm"))
        r = ell.get("radius")
        if r not in (None, "") and mm_px:
            return _round(float(r) * 2.0 * mm_px)
        return ""

    rows: List[Dict[str, Any]] = []
    for r in results_history:
        pupil = r.get("pupil", {})
        limbus = r.get("limbus", {})
        pe = pupil.get("ellipse", {})
        le = limbus.get("ellipse", {})
        cc = r.get("corneal_center", {})
        meta = r.get("metadata", {})
        cal_info = r.get("calibration", {})
        mm_px = float(cal_info.get("mm_per_px", 0) or 0)

        pupil_dia_px = (pe.get("radius", 0) or 0) * 2 if pe.get("radius") else ""
        limbus_dia_px = (le.get("radius", 0) or 0) * 2 if le.get("radius") else ""

        rows.append(
            {
                "frame": meta.get("frame_number", ""),
                "source": meta.get("source", ""),
                "processing_time_ms": _round(meta.get("processing_time_ms", ""), 2),
                "latency_ms": _round(meta.get("latency_ms", ""), 2),
                "pupil_detected": pupil.get("detected", False),
                "pupil_cx_px": _round(pe.get("center_x", "")),
                "pupil_cy_px": _round(pe.get("center_y", "")),
                "pupil_diameter_px": _round(pupil_dia_px),
                "pupil_diameter_mm": _diam_mm(pe, mm_px),
                "pupil_radius_mm": _round(pupil.get("radius_mm", "")),
                "pupil_semi_major_px": _round(pe.get("semi_major", "")),
                "pupil_semi_minor_px": _round(pe.get("semi_minor", "")),
                "pupil_semi_major_mm": _round(pe.get("semi_major_mm", "")),
                "pupil_semi_minor_mm": _round(pe.get("semi_minor_mm", "")),
                "pupil_angle_deg": _round(pe.get("angle_deg", ""), 2),
                "pupil_fit_type": pupil.get("fit_type", "") or "",
                "pupil_confidence": _round(pupil.get("confidence", ""), 3),
                "limbus_detected": limbus.get("detected", False),
                "limbus_cx_px": _round(le.get("center_x", "")),
                "limbus_cy_px": _round(le.get("center_y", "")),
                "limbus_diameter_px": _round(limbus_dia_px),
                "limbus_diameter_mm": _diam_mm(le, mm_px),
                "limbus_radius_mm": _round(limbus.get("radius_mm", "")),
                "limbus_semi_major_px": _round(le.get("semi_major", "")),
                "limbus_semi_minor_px": _round(le.get("semi_minor", "")),
                "limbus_semi_major_mm": _round(le.get("semi_major_mm", "")),
                "limbus_semi_minor_mm": _round(le.get("semi_minor_mm", "")),
                "limbus_angle_deg": _round(le.get("angle_deg", ""), 2),
                "limbus_fit_type": limbus.get("fit_type", "") or "",
                "limbus_confidence": _round(limbus.get("confidence", ""), 3),
                "corneal_center_x_px": _round(
                    (cc.get("center_px") or ["", ""])[0]
                ),
                "corneal_center_y_px": _round(
                    (cc.get("center_px") or ["", ""])[1]
                ),
                "offset_px": _round(cc.get("offset_magnitude_px", "")),
                "offset_mm": _round(cc.get("offset_magnitude_mm", "")),
                "offset_angle_deg": _round(cc.get("offset_angle_deg", ""), 2),
                "corneal_reference_source": r.get("corneal_reference_source", ""),
                "ring_status": r.get("ring_status", ""),
                "ring_center_x": _round(r.get("ring_center_x", "")),
                "ring_center_y": _round(r.get("ring_center_y", "")),
                "ring_radius_px": _round(r.get("ring_radius", "")),
                "ring_diameter_mm": (
                    _round((r.get("ring_radius") or 0) * 2.0 * mm_px)
                    if r.get("ring_radius") and mm_px else ""
                ),
                "ring_dot_count": r.get("ring_dot_count", ""),
                "calibrated": cal_info.get("calibrated", False),
                "px_per_mm": _round(cal_info.get("px_per_mm", "")),
                "mm_per_px": _round(cal_info.get("mm_per_px", ""), 6),
                "quality": r.get("overall_quality", ""),
                "overall_confidence": _round(r.get("overall_confidence", ""), 3),
                "grayscale_mode": r.get("grayscale_mode", ""),
            }
        )

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    status_var.set(f"Exported {len(rows)} rows \u2192 {path}")
    return path


# ── JSON export ────────────────────────────────────────────────────


def export_json(
    results_history: List[Dict[str, Any]],
    status_var: Any,
) -> Optional[str]:
    """Export results history as JSON. Returns the path on success, None on cancel."""
    from tkinter import filedialog, messagebox

    if not results_history:
        messagebox.showinfo("No Data", "No results to export")
        return None

    path = filedialog.asksaveasfilename(
        title="Export JSON",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
    )
    if not path:
        return None

    export_payload = {
        "export_info": {
            "version": "2.3",
            "total_frames": len(results_history),
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corneal_diameter_assumption_mm": _CORNEAL_DIAMETER_MM,
        },
        "results": results_history,
    }
    with open(path, "w") as fh:
        json.dump(export_payload, fh, indent=2, default=str)

    status_var.set(f"Exported {len(results_history)} results \u2192 {path}")
    return path


# ── Snapshot export ────────────────────────────────────────────────


def export_snapshot(
    current_image: Any,
    current_result: Any,
    prepare_frame_fn: Any,
    status_var: Any,
) -> Optional[str]:
    """Save current displayed image as PNG/JPEG. Returns path on success."""
    from tkinter import filedialog, messagebox

    if current_image is None:
        messagebox.showinfo("No Image", "No image to export")
        return None

    path = filedialog.asksaveasfilename(
        title="Save Snapshot",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
    )
    if not path:
        return None

    image = prepare_frame_fn(current_image, current_result)
    cv2.imwrite(path, image)
    status_var.set(f"Snapshot saved \u2192 {path}")
    return path

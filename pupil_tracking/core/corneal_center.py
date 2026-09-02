"""
Smoothed state writer for EyeDetectionResult.

Writes Kalman-filtered values back into typed detection results,
recalculating derived measurements (mm values, corneal centre)
to maintain consistency after smoothing.

Key correction: Corneal centre = limbus centre (anatomical definition),
NOT the midpoint of pupil and limbus centres.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pupil_tracking.utils.types import (
        EyeDetectionResult,
        CalibrationInfo,
        CornealCenterResult,
        PupilDetection,
        LimbusDetection,
        EllipseParams,
    )

# Import CornealCenterResult at runtime for the calculator class
from pupil_tracking.utils.types import CornealCenterResult


# ════════════════════════════════════════════════════════════════
# Corneal Center Calculator (minimal wrapper)
# ════════════════════════════════════════════════════════════════

class CornealCenterCalculator:
    """Minimal wrapper for corneal center calculation.
    
    Computes corneal centre as the limbus centre, and calculates
    the pupil-limbus offset. Works with EyeDetectionResult objects
    containing pupil and limbus detections.
    
    Parameters
    ----------
    config : Any, optional
        Configuration object (not currently used, for API compatibility).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def calculate(
        self,
        pupil: Any,  # PupilDetection
        limbus: Any,  # LimbusDetection
        calibration: "CalibrationInfo",
    ) -> "CornealCenterResult":
        """Calculate corneal centre and pupil-limbus offset.

        The corneal centre is defined as the limbus centre (anatomical
        definition). The offset is the displacement from corneal centre
        to pupil centre.

        Parameters
        ----------
        pupil : PupilDetection
            Pupil detection result with ellipse geometry.
        limbus : LimbusDetection
            Limbus detection result with ellipse geometry.
        calibration : CalibrationInfo
            Pixel-to-mm calibration information.

        Returns
        -------
        CornealCenterResult
            Computed corneal centre, offset, and confidence.
        """
        result = CornealCenterResult()

        # Check if both detections are present
        if not (pupil.detected and limbus.detected and 
                pupil.ellipse is not None and limbus.ellipse is not None):
            result.valid = False
            return result

        p_ell = pupil.ellipse
        l_ell = limbus.ellipse

        # Corneal centre = limbus centre (anatomical definition)
        cx = l_ell.center_x
        cy = l_ell.center_y
        result.center_px = (cx, cy)

        # Offset = pupil centre - limbus centre
        ox = p_ell.center_x - l_ell.center_x
        oy = p_ell.center_y - l_ell.center_y
        result.offset_px = (ox, oy)

        # Offset magnitude in pixels
        result.offset_magnitude_px = math.sqrt(ox * ox + oy * oy)

        # Offset angle in degrees (signed)
        result.offset_angle_deg = math.degrees(math.atan2(oy, ox))

        # Confidence based on constituent detections
        base_conf = min(pupil.confidence, limbus.confidence) * 0.8

        # Apply penalty if offset exceeds 20% of limbus radius
        if l_ell.radius > 1e-6:
            offset_ratio = result.offset_magnitude_px / l_ell.radius
            if offset_ratio > 0.2:
                base_conf *= max(
                    0.3,
                    1.0 - (offset_ratio - 0.2) * 2.0,
                )

        result.confidence = float(np.clip(base_conf, 0.0, 1.0))
        result.valid = True

        # Convert to mm if calibration is available
        if calibration is not None and calibration.calibrated:
            result.center_mm = calibration.point_px_to_mm((cx, cy))
            
            # Offset in mm (direct multiplication, not px_to_mm point conversion)
            dx_mm = ox * calibration.mm_per_px
            dy_mm = oy * calibration.mm_per_px
            result.offset_mm = (dx_mm, dy_mm)
            result.offset_magnitude_mm = math.sqrt(
                dx_mm * dx_mm + dy_mm * dy_mm
            )

        return result

    def reset(self) -> None:
        """Reset calculator state.
        
        This is a no-op in the current implementation but is kept
        for API compatibility with previous versions.
        """
        pass
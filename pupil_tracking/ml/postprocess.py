"""
Mask → contour → ellipse post-processing utilities.

These functions are used by both the inference engine and the
classical detector to convert binary masks into geometric
measurements.

Supports both 3-class and 4-class segmentation outputs:
    3-class (legacy):      0=background  1=pupil  2=iris
    4-class (ring-aware):  0=background  1=pupil  2=iris  3=suction_ring

When a 4-class prediction is available, the ring mask (class 3) is
used to:
  - Extract ring geometry (centre, radius) for spatial constraints.
  - Mask out ring pixels before pupil / iris contour extraction.
  - Validate that pupil and limbus detections lie inside the ring
    opening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.types import EllipseParams, FitResult, ANATOMICAL_LIMITS
from pupil_tracking.core.ellipse_fitter import EllipseFitter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Ring Extraction Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RingSegmentationResult:
    """Result of extracting ring geometry from a segmentation mask."""

    detected: bool = False
    """Whether a ring region was found in the segmentation output."""

    center: Optional[Tuple[float, float]] = None
    """(x, y) centre of the ring region in original image coordinates."""

    radius: Optional[float] = None
    """Equivalent circular radius of the ring in original image coords."""

    inner_radius: Optional[float] = None
    """Estimated radius of the ring opening (inner edge)."""

    mask: Optional[np.ndarray] = None
    """Binary mask of the ring class (uint8, 255 = ring pixel)."""

    area_fraction: float = 0.0
    """Fraction of the image covered by ring pixels."""

    contour: Optional[np.ndarray] = None
    """Largest contour of the ring region."""


# ═══════════════════════════════════════════════════════════════════════
#  Core Post-Processing Functions
# ═══════════════════════════════════════════════════════════════════════

def mask_to_contours(
    binary_mask: np.ndarray,
    min_area: int = 100,
    max_contours: int = 5,
) -> List[np.ndarray]:
    """Extract contours from a binary mask, sorted by area (largest first).

    Parameters
    ----------
    binary_mask : np.ndarray  uint8  (H, W)  values 0 or 255
    min_area : int  minimum contour area in pixels²
    max_contours : int  maximum number of contours to return

    Returns
    -------
    list of np.ndarray  each shape (N, 1, 2)
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # filter by area and sort descending
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid[:max_contours]


def extract_contours_from_mask(
    prediction: np.ndarray,
    class_id: int,
    min_area: int = 100,
    max_contours: int = 5,
) -> List[np.ndarray]:
    """Extract contours for a specific class from a multi-class prediction.

    Parameters
    ----------
    prediction : np.ndarray
        Class-index prediction ``(H, W)``.
    class_id : int
        Target class to extract contours for.
    min_area : int
        Minimum contour area.
    max_contours : int
        Maximum number of contours to return.

    Returns
    -------
    list of np.ndarray
        Contour arrays sorted by area (largest first).
    """
    binary = ((prediction == class_id).astype(np.uint8)) * 255
    return mask_to_contours(binary, min_area=min_area, max_contours=max_contours)


# ═══════════════════════════════════════════════════════════════════════
#  Ring Extraction from Segmentation
# ═══════════════════════════════════════════════════════════════════════

def extract_ring_from_segmentation(
    prediction: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    min_ring_area: int = 500,
) -> RingSegmentationResult:
    """Extract ring geometry from a 4-class segmentation prediction.

    Looks for class-3 (suction_ring) pixels and, if found, computes
    the ring centre, radius, and inner opening radius.

    Parameters
    ----------
    prediction : np.ndarray
        Class-index prediction ``(H, W)`` with values 0–3.
    scale_x, scale_y : float
        Scale factors to convert from model-resolution coordinates
        to original image coordinates.
    min_ring_area : int
        Minimum number of ring pixels to consider the ring detected.

    Returns
    -------
    RingSegmentationResult
        Ring geometry extracted from the segmentation mask.
    """
    ring_mask = ((prediction == 3).astype(np.uint8)) * 255
    ring_pixels = np.count_nonzero(ring_mask)

    if ring_pixels < min_ring_area:
        return RingSegmentationResult(detected=False)

    # Find the largest ring contour
    contours = mask_to_contours(ring_mask, min_area=min_ring_area, max_contours=1)
    if not contours:
        return RingSegmentationResult(detected=False)

    ring_contour = contours[0]
    ring_area = cv2.contourArea(ring_contour)

    # Compute centroid
    M = cv2.moments(ring_contour)
    if M["m00"] == 0:
        return RingSegmentationResult(detected=False)

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    # Compute equivalent radius
    equiv_radius = float(np.sqrt(ring_area / np.pi))

    # Estimate inner radius
    # The ring is an annulus — the inner opening is where there are
    # NO ring pixels inside the ring contour.
    # Use the minimum enclosing circle and subtract the ring thickness.
    (enc_cx, enc_cy), enc_radius = cv2.minEnclosingCircle(ring_contour)

    # The inner radius is approximately the outer radius minus the
    # average radial thickness of the ring mask.
    # Estimate thickness by sampling radial profiles.
    inner_radius = _estimate_ring_inner_radius(
        ring_mask, int(cx), int(cy), int(enc_radius),
    )

    # Scale to original image coordinates
    scale_r = max(scale_x, scale_y)
    total_pixels = ring_mask.shape[0] * ring_mask.shape[1]

    return RingSegmentationResult(
        detected=True,
        center=(cx * scale_x, cy * scale_y),
        radius=enc_radius * scale_r,
        inner_radius=inner_radius * scale_r if inner_radius else None,
        mask=ring_mask,
        area_fraction=ring_pixels / max(total_pixels, 1),
        contour=ring_contour,
    )


def _estimate_ring_inner_radius(
    ring_mask: np.ndarray,
    cx: int, cy: int,
    outer_radius: int,
    num_rays: int = 36,
) -> Optional[float]:
    """Estimate the inner radius of a ring annulus from its mask.

    Casts radial rays from the centre outward and finds where the
    ring mask transitions from 0 (inner opening) to 255 (ring).
    The average transition distance is the inner radius.

    Parameters
    ----------
    ring_mask : np.ndarray
        Binary ring mask (uint8, 0 or 255).
    cx, cy : int
        Centre of the ring in mask coordinates.
    outer_radius : int
        Approximate outer radius of the ring.
    num_rays : int
        Number of radial rays to cast.

    Returns
    -------
    float or None
        Estimated inner radius, or None if estimation fails.
    """
    h, w = ring_mask.shape[:2]
    inner_distances = []

    for i in range(num_rays):
        angle = 2.0 * np.pi * i / num_rays

        for r in range(1, outer_radius):
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))

            if 0 <= px < w and 0 <= py < h:
                if ring_mask[py, px] > 127:
                    # Found the inner edge of the ring
                    inner_distances.append(float(r))
                    break

    if len(inner_distances) < num_rays * 0.3:
        return None

    return float(np.median(inner_distances))


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_pupil_limbus_pair(
    pupil: Optional[EllipseParams],
    limbus: Optional[EllipseParams],
    ring: Optional[RingSegmentationResult] = None,
) -> Tuple[bool, List[str]]:
    """Cross-validate pupil and limbus detections.

    When a ring detection is available, additionally checks that
    both the pupil and limbus lie inside the ring opening.

    Parameters
    ----------
    pupil : EllipseParams or None
        Pupil detection result.
    limbus : EllipseParams or None
        Limbus detection result.
    ring : RingSegmentationResult or None
        Ring detection result from segmentation.

    Returns
    -------
    (valid, list_of_issues)
        ``valid`` is True if no issues were found.
    """
    issues: List[str] = []
    limits = ANATOMICAL_LIMITS

    if pupil is None or limbus is None:
        return True, issues  # can't cross-validate without both

    if not pupil.is_valid or not limbus.is_valid:
        return True, issues

    # ── Pupil / limbus ratio check ────────────────────────────
    if limbus.radius > 0:
        ratio = pupil.radius / limbus.radius
        if ratio < limits.MIN_PUPIL_LIMBUS_RATIO:
            issues.append(
                f"Pupil/limbus ratio too small: {ratio:.3f} "
                f"(min={limits.MIN_PUPIL_LIMBUS_RATIO})"
            )
        if ratio > limits.MAX_PUPIL_LIMBUS_RATIO:
            issues.append(
                f"Pupil/limbus ratio too large: {ratio:.3f} "
                f"(max={limits.MAX_PUPIL_LIMBUS_RATIO})"
            )

    # ── Containment: pupil center inside limbus ───────────────
    dx = pupil.center_x - limbus.center_x
    dy = pupil.center_y - limbus.center_y
    offset = (dx ** 2 + dy ** 2) ** 0.5
    if limbus.radius > 0:
        offset_ratio = offset / limbus.radius
        if offset_ratio > limits.MAX_CENTER_OFFSET_RATIO:
            issues.append(
                f"Pupil center offset too large: {offset_ratio:.3f} "
                f"of limbus radius (max={limits.MAX_CENTER_OFFSET_RATIO})"
            )

    # ── Ring containment checks ───────────────────────────────
    if ring is not None and ring.detected and ring.center is not None:
        ring_cx, ring_cy = ring.center
        ring_r = ring.inner_radius or (
            ring.radius * 0.80 if ring.radius else None
        )

        if ring_r is not None:
            # Pupil centre should be inside ring opening
            pupil_dist = np.sqrt(
                (pupil.center_x - ring_cx) ** 2
                + (pupil.center_y - ring_cy) ** 2
            )
            if pupil_dist > ring_r * 0.85:
                issues.append(
                    f"Pupil centre is near or outside ring opening: "
                    f"dist={pupil_dist:.1f}, ring_inner_r={ring_r:.1f}"
                )

            # Limbus should be inside ring opening
            limbus_dist = np.sqrt(
                (limbus.center_x - ring_cx) ** 2
                + (limbus.center_y - ring_cy) ** 2
            )
            if limbus.radius > 0 and (limbus_dist + limbus.radius) > ring_r * 1.1:
                issues.append(
                    f"Limbus extends outside ring opening: "
                    f"limbus edge at {limbus_dist + limbus.radius:.1f}, "
                    f"ring_inner_r={ring_r:.1f}"
                )

            # Limbus radius should be smaller than ring
            if ring.radius is not None and limbus.radius > ring.radius * 0.95:
                issues.append(
                    f"Limbus radius ({limbus.radius:.1f}) is nearly as "
                    f"large as ring radius ({ring.radius:.1f}) — "
                    f"possibly detecting ring as limbus"
                )

    valid = len(issues) == 0
    return valid, issues


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _scale_fit_to_ellipse(
    fit: FitResult,
    scale_x: float,
    scale_y: float,
) -> EllipseParams:
    """Scale a FitResult from model space to original image space."""
    scale_r = max(scale_x, scale_y)
    scale_r_min = min(scale_x, scale_y)

    return EllipseParams(
        center_x=fit.center_x * scale_x,
        center_y=fit.center_y * scale_y,
        semi_major=fit.semi_major * scale_r,
        semi_minor=fit.semi_minor * scale_r_min,
        angle_deg=fit.angle_deg,
        uncertainty_center_x=fit.uncertainty_center[0] * scale_x,
        uncertainty_center_y=fit.uncertainty_center[1] * scale_y,
        uncertainty_semi_major=fit.uncertainty_radius * scale_r,
        uncertainty_semi_minor=fit.uncertainty_radius * scale_r_min,
        fit_quality=fit.fit_quality_score,
        fit_rms_residual=fit.rms_residual * scale_r,
        num_contour_points=fit.num_points,
        eccentricity=fit.eccentricity,
        circularity=(
            fit.semi_minor / fit.semi_major
            if fit.semi_major > 0 else 0.0
        ),
    )
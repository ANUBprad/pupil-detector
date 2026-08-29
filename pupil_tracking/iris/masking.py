"""Reflection / occlusion masking for the iris region.

The mask marks which iris-annulus pixels are *usable* (not specular-reflective
and not pupil/sclera). Reflection detection reuses the existing
``ReflectionRemover`` so we do not introduce a redundant reflection
implementation.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from pupil_tracking.iris.roi import sample_annulus_mask
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover


class IrisMasking:
    """Build a usable-pixels mask for the iris annulus.

    Parameters
    ----------
    reflection_remover : ReflectionRemover | None
        Existing reflection detector. If None, a default ``ReflectionRemover``
        is constructed (using the existing implementation).
    only_within_roi : bool
        If True, reflections outside the annulus are ignored (they do not
        affect the iris region).
    """

    def __init__(
        self,
        reflection_remover: Optional[ReflectionRemover] = None,
        only_within_roi: bool = True,
    ) -> None:
        self.reflection_remover = reflection_remover or ReflectionRemover()
        self.only_within_roi = only_within_roi

    def build(
        self,
        image_bgr: np.ndarray,
        roi: IrisROI,
        *,
        external_occlusion: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a boolean (H, W) mask of *usable* iris pixels.

        A pixel is usable when it lies inside the annulus, is not a specular
        reflection, and is not excluded by an external occlusion mask.

        Parameters
        ----------
        image_bgr : np.ndarray  (H, W, 3) BGR image
        roi : IrisROI
        external_occlusion : np.ndarray | None
            Optional boolean (H, W) mask where True = occluded (e.g. eyelids).
        """
        if image_bgr is None:
            return np.zeros((0, 0), dtype=bool)
        h, w = image_bgr.shape[:2]

        annulus = sample_annulus_mask((h, w), roi)

        # Reflection detection, restricted to the annulus when requested.
        if self.only_within_roi:
            roi_255 = annulus.astype(np.uint8) * 255
            _, reflection = self.reflection_remover.remove(
                image_bgr, roi_mask=roi_255
            )
        else:
            _, reflection = self.reflection_remover.remove(image_bgr)
        reflection = reflection > 0

        usable = annulus & ~reflection

        if external_occlusion is not None and external_occlusion.shape == (h, w):
            usable = usable & ~external_occlusion.astype(bool)

        return usable


def mask_stats(usable: np.ndarray, roi: IrisROI) -> dict:
    """Return compact statistics about the usable mask.

    Values are reported honestly: when the mask is empty or the ROI invalid,
    the fractions are 0 rather than inflated.
    """
    if usable is None or usable.size == 0 or not roi.valid:
        return {
            "usable_iris_pixels": 0,
            "usable_fraction": 0.0,
            "annulus_area_px": 0.0,
            "reflection_estimated": False,
        }
    n_usable = int(np.count_nonzero(usable))
    # Estimate total annulus area from mean radii (approximate, for coverage).
    annulus_area = (
        np.pi * (roi.limbus_radius_px ** 2 - roi.pupil_radius_px ** 2)
    )
    annulus_area = max(annulus_area, 1.0)
    return {
        "usable_iris_pixels": n_usable,
        "usable_fraction": float(n_usable / annulus_area),
        "annulus_area_px": float(annulus_area),
        "reflection_estimated": True,
    }

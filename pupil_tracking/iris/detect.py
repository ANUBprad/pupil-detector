"""Top-level iris-feature detection orchestration (Phase I).

This is the public entry point that wires together:

    ROI construction -> masking -> normalization -> extraction -> result

It consumes the existing pupil/limbus geometry (``EllipseParams``) and does
**not** re-run pupil/limbus detection. It is safe to call with missing or
invalid geometry (returns a non-crashing ``IrisDetectionResult`` with
``status=NO_ROI``).

The iris detector is **disabled by default** in the production pipeline; it is
invoked explicitly when needed. No matching, registration, rotation estimation
or cyclotorsion logic is present here.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.extraction import IrisFeatureExtractor
from pupil_tracking.iris.masking import IrisMasking, mask_stats
from pupil_tracking.iris.roi import IrisROIExtractor
from pupil_tracking.iris.types import IrisDetectionResult, IrisFeatureSet, IrisStatus
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
from pupil_tracking.utils.types import EllipseParams


class IrisFeatureDetector:
    """Stateful detector wrapping ROI, masking and extraction.

    Instantiate once and call :meth:`detect` per image. All parameters are read
    from :class:`IrisConfig`.
    """

    def __init__(
        self,
        config: Optional[IrisConfig] = None,
        reflection_remover: Optional[ReflectionRemover] = None,
    ) -> None:
        self.config = config or IrisConfig()
        self.extractor = IrisFeatureExtractor(
            num_angles=self.config.num_angles,
            num_radii=self.config.num_radii,
            radius_px=self.config.radius_px,
            min_contrast=self.config.min_contrast,
            max_features=self.config.max_features,
            min_angular_sep_deg=self.config.min_angular_sep_deg,
        )
        self.roi_extractor = IrisROIExtractor(
            inner_inset_frac=self.config.inner_inset_frac,
            outer_inset_frac=self.config.outer_inset_frac,
        )
        self.masking = IrisMasking(
            reflection_remover=reflection_remover,
            only_within_roi=self.config.only_within_roi,
        )

    def detect(
        self,
        image: np.ndarray,
        pupil: Optional[EllipseParams],
        limbus: Optional[EllipseParams],
        *,
        external_occlusion: Optional[np.ndarray] = None,
    ) -> IrisDetectionResult:
        """Run iris-feature detection.

        Parameters
        ----------
        image : np.ndarray  BGR (H, W, 3) or grayscale (H, W)
        pupil : EllipseParams | None
        limbus : EllipseParams | None
        external_occlusion : np.ndarray | None
            Optional boolean (H, W) mask of occluded pixels (True = occluded).

        Returns
        -------
        IrisDetectionResult
        """
        start = time.perf_counter()

        roi = self.roi_extractor.build(pupil, limbus)
        if not roi.valid:
            return self._finish(
                IrisDetectionResult(
                    valid=False,
                    status=IrisStatus.NO_ROI,
                    feature_set=IrisFeatureSet(roi=roi),
                ),
                start,
            )

        usable = self.masking.build(image, roi, external_occlusion=external_occlusion)

        feature_set = self.extractor.extract(
            image,
            roi,
            usable,
            pupil=pupil,
            limbus=limbus,
        )
        feature_set.usable_fraction = mask_stats(usable, roi).get(
            "usable_fraction", 0.0
        )
        feature_set.region_coverage = self._coverage(feature_set, roi)

        n_features = len(feature_set.features)
        status = (
            IrisStatus.OK
            if n_features > 0
            else IrisStatus.NO_FEATURES
        )

        result = IrisDetectionResult(
            valid=n_features > 0,
            status=status,
            feature_set=feature_set,
            mask_stats=mask_stats(usable, roi),
        )
        return self._finish(result, start)

    def detect_from_ellipses(self, image, pupil, limbus, **kwargs):
        """Alias for :meth:`detect` (explicit geometry entry point)."""
        return self.detect(image, pupil, limbus, **kwargs)

    @staticmethod
    def _coverage(feature_set: IrisFeatureSet, roi) -> float:
        """Fraction of the annulus (coarsely) 'covered' by accepted features.

        Cover is estimated as the summed area of the accepted feature patches
        relative to the annulus area. It is an upper-bound style indicator, not
        a precise geometric coverage; interpreted only as a coarse metric.
        """
        if not roi.valid or roi.limbus_radius_px <= roi.pupil_radius_px:
            return 0.0
        annulus_area = np.pi * (
            roi.limbus_radius_px ** 2 - roi.pupil_radius_px ** 2
        )
        if annulus_area <= 0:
            return 0.0
        n = len(feature_set.features)
        # Approximate each accepted feature as covering a small disc of radius
        # 3 px. Not a precise geometric coverage; used only as a coarse
        # distribution indicator.
        r = 3.0
        area_sum = n * (np.pi * r * r)
        return float(min(area_sum / annulus_area, 1.0))

    @staticmethod
    def _finish(result: IrisDetectionResult, start: float) -> IrisDetectionResult:
        result.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return result


def detect_iris_features(
    image: np.ndarray,
    pupil: Optional[EllipseParams],
    limbus: Optional[EllipseParams],
    *,
    config: Optional[IrisConfig] = None,
    reflection_remover: Optional[ReflectionRemover] = None,
    external_occlusion: Optional[np.ndarray] = None,
) -> IrisDetectionResult:
    """Convenience one-shot iris-feature detection.

    Equivalent to constructing an :class:`IrisFeatureDetector` and calling
    :meth:`IrisFeatureDetector.detect`.
    """
    detector = IrisFeatureDetector(
        config=config,
        reflection_remover=reflection_remover,
    )
    return detector.detect(
        image,
        pupil,
        limbus,
        external_occlusion=external_occlusion,
    )

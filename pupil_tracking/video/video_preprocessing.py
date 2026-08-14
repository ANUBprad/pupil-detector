# pupil_tracking/video/video_preprocessing.py
"""Video preprocessing and frame quality checking extracted from
:class:`OptimizedVideoProcessor` during the Phase-5 refactoring.

These are fully independent classes that only depend on OpenCV and
internal preprocessing modules.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """ACCURACY-FIRST preprocessing for video frames.

    Ensures consistent normalisation regardless of illumination.
    Includes reflection removal, suction ring masking, and CLAHE.

    Plan alignment:
        A3 - reflection removal
        A5 - suction ring marker masking
        A6 - ImageNormalizer for brightness/contrast consistency
        S4 - bilateral filter removed from fast path
    """

    def __init__(
        self,
        denoise_strength: int = 3,
        clahe_clip: float = 2.0,
        clahe_grid: int = 4,
        sharpen: bool = False,
        fast_mode: bool = True,
        apply_normalizer: bool = True,
        suction_ring_removal: bool = True,
    ):
        self.denoise_strength = denoise_strength
        self.sharpen = sharpen
        self.fast_mode = fast_mode

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )

        from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover

        self._reflection_remover = ReflectionRemover(
            brightness_threshold=225,
            min_reflection_area=10,
            inpaint_radius=3,
            detect_red_highlights=True,
            red_threshold_offset=25,
        )

        try:
            from pupil_tracking.preprocessing.temporal_reflection_filter import (
                TemporalReflectionFilter,
            )

            self._temporal_filter = TemporalReflectionFilter(
                history_size=5,
                blink_threshold=0.3,
                min_stable_frames=2,
                dilation_size=3,
            )
        except ImportError:
            self._temporal_filter = None
            logger.debug("TemporalReflectionFilter not available")

        self._ring_masker = None
        if suction_ring_removal:
            from pupil_tracking.preprocessing.suction_ring_masker import (
                SuctionRingMasker,
            )

            self._ring_masker = SuctionRingMasker()

        self._normalizer = None
        if apply_normalizer:
            try:
                from pupil_tracking.preprocessing.normalizer import ImageNormalizer

                self._normalizer = ImageNormalizer(
                    enable_clahe=True,
                    enable_brightness=True,
                    enable_white_balance=False,
                    enable_gamma=False,
                )
                logger.info("ImageNormalizer enabled for video preprocessing")
            except ImportError:
                logger.warning(
                    "ImageNormalizer not available - falling back to CLAHE only"
                )

        self._current_stable_mask: Optional[np.ndarray] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply video-optimised preprocessing with proper normalisation."""
        if image is None or image.size == 0:
            return image

        out = image

        if self._ring_masker is not None:
            try:
                out, _ = self._ring_masker.remove(out)
            except Exception:
                pass

        out, _ = self._reflection_remover.remove(out, roi_mask=None)

        if self._temporal_filter is not None:
            stable_mask = self._temporal_filter.process(out)
            self._current_stable_mask = stable_mask

        if self._normalizer is not None:
            out = self._normalizer.normalize(out)
        else:
            if len(out.shape) == 3 and out.shape[2] >= 3:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif len(out.shape) == 2:
                out = self._clahe.apply(out)

        if self.fast_mode:
            return out

        if self.denoise_strength > 0:
            out = cv2.bilateralFilter(
                out,
                d=self.denoise_strength,
                sigmaColor=35,
                sigmaSpace=35,
            )

        if self.sharpen:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
            out = cv2.addWeighted(out, 1.2, blurred, -0.2, 0)

        return out


class FrameQualityChecker:
    """Quality checker with permissive thresholds for surgical images."""

    def __init__(
        self,
        blur_threshold: float = 20.0,
        brightness_low: float = 15.0,
        brightness_high: float = 250.0,
        skip_check: bool = False,
    ):
        self.blur_thresh = blur_threshold
        self.bright_lo = brightness_low
        self.bright_hi = brightness_high
        self.skip_check = skip_check

    def is_usable(self, image: np.ndarray) -> Tuple[bool, str]:
        """Returns (usable, reason). ~0.1ms if skip_check=True."""
        if image is None or image.size == 0:
            return False, "empty_frame"

        if self.skip_check:
            return True, "ok"

        gray = (
            image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        mean_bright = float(gray.mean())
        if mean_bright < self.bright_lo:
            return False, "too_dark"
        if mean_bright > self.bright_hi:
            return False, "too_bright"

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.blur_thresh:
            return False, "too_blurry"

        return True, "ok"

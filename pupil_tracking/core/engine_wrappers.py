# pupil_tracking/core/engine_wrappers.py
"""ML engine wrappers that normalise the ONNX and dummy backends
to match the ``SegmentationInference`` interface expected by
:class:`UnifiedDetector`.

Extracted from detector.py during the Phase-3 refactoring to keep
the main detector module focused on detection orchestration.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from pupil_tracking.utils.types import EyeDetectionResult, FrameMetadata

logger = logging.getLogger(__name__)


class ONNXEngineWrapper:
    """Wraps ``ONNXInference`` to match the ``SegmentationInference`` interface
    that the rest of ``UnifiedDetector`` expects.

    ``SegmentationInference`` has:
      - .detect(image, frame_number, source) -> EyeDetectionResult
      - .available (bool)
      - .set_red_light_filter_enabled(bool)
      - .set_red_light_temporal_mode(bool)
      - .reset_red_light_temporal()
      - .model_path (str)

    ``ONNXInference`` has:
      - .infer(image, target_size) -> dict of masks
      - .is_loaded (bool)
    """

    def __init__(self, onnx_engine, config=None):
        self._engine = onnx_engine
        self._config = config
        self.available = onnx_engine.is_loaded
        self.model_path = None  # No .pth path for ONNX

        from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
        from pupil_tracking.preprocessing.suction_ring_masker import SuctionRingMasker

        self._reflection_remover = ReflectionRemover(
            brightness_threshold=220,
            min_reflection_area=15,
            inpaint_radius=5,
            detect_red_highlights=True,
            red_threshold_offset=25,
        )
        self._ring_masker = SuctionRingMasker()
        self._red_light_filter = None  # Lazy initialization
        self._red_light_enabled = True
        self._red_light_temporal_mode = True

    def _get_red_light_filter(self):
        """Lazy import and return red light filter."""
        from pupil_tracking.preprocessing.red_light_filter import RedLightFilter

        return RedLightFilter(
            red_threshold=200,
            dominance_offset=30,
            min_area=5,
            enable_inpaint=True,
            inpaint_radius=3,
            enable_temporal=self._red_light_temporal_mode,
        )

    def detect(
        self,
        image: np.ndarray,
        frame_number: int = -1,
        source: str = "",
        **kwargs,
    ) -> EyeDetectionResult:
        """Run ONNX inference and return an EyeDetectionResult
        with ``_raw_mask`` attached for SmartContourFitter.
        """
        clean_bgr = image
        roi_mask = None
        if self._ring_masker is not None:
            try:
                clean_bgr, marker_mask, ring_result = self._ring_masker.remove_with_diagnostics(clean_bgr)
                if getattr(ring_result, "ring_centre", None) is not None:
                    cx, cy = int(round(ring_result.ring_centre[0])), int(round(ring_result.ring_centre[1]))
                    inner_r = int(round(ring_result.ring_inner_radius)) if getattr(ring_result, "ring_inner_radius", None) is not None else None
                    if inner_r is not None and inner_r > 4:
                        h, w = clean_bgr.shape[:2]
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.circle(roi_mask, (cx, cy), inner_r, 255, -1)
            except Exception as e:
                logger.debug("Ring masker diagnostics failed, using basic removal: %s", e)
                try:
                    clean_bgr, _ = self._ring_masker.remove(clean_bgr)
                except Exception as e2:
                    logger.warning("Ring masker basic removal also failed: %s", e2)

        if self._reflection_remover is not None:
            clean_bgr, _ = self._reflection_remover.remove(clean_bgr, roi_mask=roi_mask)

        if self._red_light_enabled:
            if self._red_light_filter is None:
                self._red_light_filter = self._get_red_light_filter()
            if self._red_light_filter is not None:
                clean_bgr, _ = self._red_light_filter.apply(
                    clean_bgr, frame_number=frame_number
                )

        masks = self._engine.infer(clean_bgr)

        result = EyeDetectionResult()
        result.metadata = FrameMetadata()
        result.metadata.frame_number = frame_number
        result.metadata.source = source

        h, w = image.shape[:2]
        raw_mask = np.zeros((h, w), dtype=np.uint8)

        iris_mask = masks.get("iris", np.zeros((h, w), dtype=np.uint8))
        pupil_mask = masks.get("pupil", np.zeros((h, w), dtype=np.uint8))
        ring_mask = masks.get("ring", None)

        raw_mask[iris_mask > 127] = 2
        raw_mask[pupil_mask > 127] = 1
        if ring_mask is not None:
            raw_mask[ring_mask > 127] = 3

        result._raw_mask = raw_mask

        pupil_pixels = (pupil_mask > 127).sum()
        iris_pixels = (iris_mask > 127).sum()

        if pupil_pixels > 100:
            result.pupil.detected = True
            result.pupil.confidence = 0.5
        if iris_pixels > 100:
            result.limbus.detected = True
            result.limbus.confidence = 0.5

        return result

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable red light filtering."""
        self._red_light_enabled = enabled
        if enabled:
            self._red_light_filter = self._get_red_light_filter()
        else:
            try:
                from pupil_tracking.preprocessing.red_light_filter import RedLightFilter
                self._red_light_filter = RedLightFilter(
                    red_threshold=255,
                    dominance_offset=1000,
                    min_area=100000,
                    enable_inpaint=False,
                    enable_temporal=False,
                )
            except Exception as e:
                logger.warning("Failed to create disabled red light filter: %s", e)
                self._red_light_filter = None

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        """Enable temporal mode for red light filtering (for video)."""
        self._red_light_temporal_mode = enabled
        if self._red_light_filter is None:
            self._red_light_filter = self._get_red_light_filter()
        if self._red_light_filter is not None:
            self._red_light_filter.enable_temporal = enabled
            if not enabled:
                self._red_light_filter.reset_temporal()

    def reset_red_light_temporal(self) -> None:
        """Reset temporal tracking state for red light filter."""
        if self._red_light_filter is not None:
            self._red_light_filter.reset_temporal()

    def get_device_info(self) -> dict:
        return self._engine.get_device_info()

    def __getattr__(self, name):
        """Forward unknown attributes to the underlying engine."""
        return getattr(self._engine, name)


class DummyEngine:
    """Fallback engine when no ML backend is available.
    Returns empty results so classical fallback can still work.
    """

    available = False
    model_path = None

    def detect(self, image: np.ndarray, **kwargs) -> EyeDetectionResult:
        result = EyeDetectionResult()
        result.metadata = FrameMetadata()
        h, w = image.shape[:2]
        result._raw_mask = np.zeros((h, w), dtype=np.uint8)
        return result

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        pass

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        pass

    def reset_red_light_temporal(self) -> None:
        pass

    def get_device_info(self) -> dict:
        return {"backend": "none", "model": "none"}

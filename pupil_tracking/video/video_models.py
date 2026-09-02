# pupil_tracking/video/video_models.py
"""Data models, constants, and utility functions extracted from
:class:`OptimizedVideoProcessor` during the Phase-5 refactoring.

These are pure data classes and functions with no dependencies
on the main processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ManualCircularROI:
    """User-defined circular ROI stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height


@dataclass
class ManualRingAnnotation:
    """User-confirmed suction-ring circle stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    dot_count: int = 12

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height

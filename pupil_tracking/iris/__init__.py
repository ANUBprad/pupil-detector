"""Iris feature detection (Phase I concept model).

This package implements the classical iris-feature-detection baseline only. It
**does not** implement feature matching, image registration, rotation/cyclotorsion
estimation, or astigmatism-axis correction -- those are later phases.

The iris detector is **disabled by default** in the production pipeline and is
invoked explicitly (e.g. via :func:`detect_iris_features`).

Additive polar registration (experimental) is available via :mod:`polar`:
geometry-aware ellipse unwrap + phase/circular angular correlation, disabled
by default (``PolarRegistrationConfig.enabled = False``) and never a source of
clinically actionable rotation.

Public API
----------
* ``IrisFeatureDetector`` / ``detect_iris_features`` -- top-level detection
* ``IrisConfig`` -- tunable parameters
* ``IrisDetectionResult`` / ``IrisFeatureSet`` / ``IrisFeature`` / ``IrisROI``
  -- result contracts
* ``draw_iris_overlay`` -- optional debug visualisation
* ``estimate_iris_rotation`` / ``detect_iris_rotation`` /
  ``estimate_polar_registration`` / ``PolarRegistrationConfig`` /
  ``PolarRegistrationResult`` -- experimental polar angular registration
"""

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.detect import IrisFeatureDetector, detect_iris_features
from pupil_tracking.iris.masking import IrisMasking
from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.polar import (
    PolarFailureKind,
    PolarRegistrationConfig,
    PolarRegistrationResult,
    circular_rotation_scan,
    detect_iris_rotation,
    estimate_iris_rotation,
    estimate_polar_registration,
    gradient_magnitude,
    normalize_polar,
    phase_rotation,
    shift_to_rotation_deg,
    unwrap_iris,
)
from pupil_tracking.iris.roi import IrisROIExtractor
from pupil_tracking.iris.types import (
    IrisDetectionResult,
    IrisFeature,
    IrisFeatureSet,
    IrisFeatureType,
    IrisROI,
    IrisStatus,
)
from pupil_tracking.iris.visualize import draw_iris_overlay

__all__ = [
    "IrisConfig",
    "IrisFeatureDetector",
    "detect_iris_features",
    "IrisMasking",
    "IrisNormalizer",
    "IrisROIExtractor",
    "IrisDetectionResult",
    "IrisFeature",
    "IrisFeatureSet",
    "IrisFeatureType",
    "IrisROI",
    "IrisStatus",
    "draw_iris_overlay",
    "PolarFailureKind",
    "PolarRegistrationConfig",
    "PolarRegistrationResult",
    "circular_rotation_scan",
    "detect_iris_rotation",
    "estimate_iris_rotation",
    "estimate_polar_registration",
    "gradient_magnitude",
    "normalize_polar",
    "phase_rotation",
    "shift_to_rotation_deg",
    "unwrap_iris",
]

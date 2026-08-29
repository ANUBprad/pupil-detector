"""Configuration for the Phase I iris-feature detector.

All tunable parameters for the concept model live in one dataclass so the
pipeline can be configured without touching call sites. Defaults are chosen
conservatively; thresholds that are not yet justified by real ELITA data are
intentionally left as configurable values to be established experimentally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IrisConfig:
    """Tunable parameters for the iris-feature concept model."""

    # ROI construction
    inner_inset_frac: float = 0.12   # back away from pupil edge (frac of pupil radius)
    outer_inset_frac: float = 0.12   # back away from limbus edge (frac of limbus radius)

    # Feature extraction lattice
    num_angles: int = 72
    num_radii: int = 8
    radius_px: int = 5

    # Feature quality filtering
    min_contrast: float = 4.0        # minimum mean texture energy (to be tuned)
    max_features: int = 120          # hard cap on accepted features
    min_angular_sep_deg: float = 5.0

    # Masking
    only_within_roi: bool = True

    # Convenience constructor / defaults are all above.
    def to_dict(self) -> dict:
        return {
            "inner_inset_frac": self.inner_inset_frac,
            "outer_inset_frac": self.outer_inset_frac,
            "num_angles": self.num_angles,
            "num_radii": self.num_radii,
            "radius_px": self.radius_px,
            "min_contrast": self.min_contrast,
            "max_features": self.max_features,
            "min_angular_sep_deg": self.min_angular_sep_deg,
            "only_within_roi": self.only_within_roi,
        }

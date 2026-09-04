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
    texture_floor: float = 2.5       # absolute minimum response gate (Daugman: protect against flat/zero-texture iris)
    texture_rel_frac: float = 0.5    # fraction of ROI texture_response_mean used for adaptive gate
    max_features: int = 120          # hard cap on accepted features
    min_angular_sep_deg: float = 5.0

    # Reflection / specular masking
    reflection_brightness_threshold: int = 230
    reflection_saturation_threshold: int = 40
    reflection_min_area: int = 20

    # Eyelid / eyelash masking (heuristic, applied only inside the annulus)
    eyelid_method: str = "gradient"   # "gradient" | "none"
    eyelid_edge_threshold: float = 60.0   # gradient magnitude -> eyelid candidate
    eyelid_dilate_px: int = 6

    # Saturation masking
    saturation_threshold: int = 250    # per-pixel brightness that is hopelessly white
    saturated_frac_limit: float = 0.5  # max accepted patch saturated fraction

    # ROI-based intensity validation (percentile scaling across acquisition types)
    use_roi_percentiles: bool = True   # derive thresholds from isolated iris ROI
    roi_p05: float = 0.05
    roi_p95: float = 0.95
    intensity_low_frac: float = 0.30   # patch mean must exceed this fraction of ROI p05->p95 span
    intensity_high_frac: float = 0.80

    # Patch-acceptance validation
    min_patch_valid_fraction: float = 0.7   # patch must be >=70% inside valid mask

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
            "texture_floor": self.texture_floor,
            "texture_rel_frac": self.texture_rel_frac,
            "max_features": self.max_features,
            "min_angular_sep_deg": self.min_angular_sep_deg,
            "reflection_brightness_threshold": self.reflection_brightness_threshold,
            "reflection_saturation_threshold": self.reflection_saturation_threshold,
            "reflection_min_area": self.reflection_min_area,
            "eyelid_method": self.eyelid_method,
            "eyelid_edge_threshold": self.eyelid_edge_threshold,
            "eyelid_dilate_px": self.eyelid_dilate_px,
            "saturation_threshold": self.saturation_threshold,
            "saturated_frac_limit": self.saturated_frac_limit,
            "use_roi_percentiles": self.use_roi_percentiles,
            "roi_p05": self.roi_p05,
            "roi_p95": self.roi_p95,
            "intensity_low_frac": self.intensity_low_frac,
            "intensity_high_frac": self.intensity_high_frac,
            "min_patch_valid_fraction": self.min_patch_valid_fraction,
            "only_within_roi": self.only_within_roi,
        }

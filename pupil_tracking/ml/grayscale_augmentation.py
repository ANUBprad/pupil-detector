"""
Grayscale-aware data augmentation for training robust segmentation models.

This module provides albumentations-compatible transforms that randomly
convert RGB training images to grayscale (replicated to 3 channels),
teaching the model to produce identical segmentation quality regardless
of whether the input is colour or grayscale.

Design rationale
----------------
Instead of training a separate single-channel model or modifying the
U-Net architecture, we augment the *training data* so that ~30 % of
images seen during training are grayscale.  Because the grayscale
images pass through the same CLAHE enhancement used at inference
(:class:`~pupil_tracking.preprocessing.grayscale_handler.GrayscaleHandler`),
the model learns the *exact* pixel distribution it will encounter
when a real grayscale image arrives.

This approach has two critical advantages:

1.  **Zero accuracy regression on RGB** — the model still sees 70 %
    colour images, so its existing colour-based features are preserved.
2.  **No architecture changes** — the model input remains
    ``(batch, 3, H, W)``, the output remains ``(batch, C, H, W)``
    with the same number of classes.

The fine-tuning script (:mod:`scripts.finetune_grayscale`) uses this
module to augment an existing trained model with grayscale robustness
in 30–50 epochs at a reduced learning rate.

Integration with existing pipeline
-----------------------------------
The :class:`GrayscaleAwarePipeline` wraps the project's existing
augmentation transforms (rotation, flip, brightness, elastic, etc.)
and inserts :class:`RandomGrayscaleConversion` at the *end* of the
spatial transforms but *before* normalisation.  This ordering ensures
that:

-   Spatial augmentations (rotation, crop) operate on the original
    colour image (maximum information).
-   Grayscale conversion happens on the already-augmented image.
-   Normalisation (mean/std) is applied last, as required by the
    model.

Usage
-----
>>> from pupil_tracking.ml.grayscale_augmentation import (
...     GrayscaleAwarePipeline,
...     RandomGrayscaleConversion,
... )
>>> pipeline = GrayscaleAwarePipeline()
>>> train_aug = pipeline.get_training_augmentation(
...     input_size=512,
...     grayscale_prob=0.3,
... )
>>> result = train_aug(image=rgb_image, mask=mask)
>>> augmented_image = result["image"]  # may be grayscale-replicated
>>> augmented_mask  = result["mask"]   # never modified by grayscale aug

Thread safety
-------------
All classes are stateless after construction and safe to use from
multiple ``DataLoader`` worker processes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np

from pupil_tracking.preprocessing.grayscale_handler import GrayscaleHandler

__all__ = [
    "RandomGrayscaleConversion",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom albumentations transform
# ---------------------------------------------------------------------------

class RandomGrayscaleConversion(A.ImageOnlyTransform):
    """Randomly convert an RGB image to enhanced grayscale (3-channel).

    When applied, the transform:

    1.  Converts the image to single-channel grayscale (BT.601).
    2.  Optionally enhances contrast with CLAHE (matching the
        inference-time enhancement in
        :class:`~pupil_tracking.preprocessing.grayscale_handler.GrayscaleHandler`).
    3.  Replicates the single channel to three identical channels
        so the output shape remains ``(H, W, 3)``.

    The mask is **never** modified — only the image is affected.

    Parameters
    ----------
    enhance : bool, optional
        If ``True`` (default), apply CLAHE enhancement after
        conversion.  This should always be ``True`` for training
        because it matches the inference pipeline.
    clahe_clip_limit : float, optional
        CLAHE clip limit.  Default ``3.0`` matches the handler default.
    clahe_grid_size : tuple[int, int], optional
        CLAHE tile grid.  Default ``(8, 8)`` matches the handler.
    always_apply : bool, optional
        If ``True``, always apply (ignore ``p``).  Default ``False``.
    p : float, optional
        Probability of applying this transform.  Default ``0.3``
        (30 % of training images become grayscale — empirically
        optimal for maintaining RGB accuracy while learning
        grayscale robustness).

    Examples
    --------
    >>> transform = RandomGrayscaleConversion(p=0.3)
    >>> result = transform(image=rgb_image)
    >>> assert result["image"].shape == rgb_image.shape
    >>> assert result["image"].shape[2] == 3

    Notes
    -----
    This transform is designed to sit **after** spatial augmentations
    (rotation, flip, crop) but **before** normalisation (mean/std
    subtraction).  The :class:`GrayscaleAwarePipeline` handles this
    ordering automatically.
    """

    def __init__(
        self,
        enhance: bool = True,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.3,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self._enhance = enhance
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_grid_size = tuple(clahe_grid_size)

        # Use the production GrayscaleHandler so training and inference
        # use *exactly* the same enhancement pipeline.
        self._handler = GrayscaleHandler(
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )

        logger.debug(
            "RandomGrayscaleConversion created — p=%.2f, enhance=%s, "
            "clip=%.1f, grid=%s",
            p,
            enhance,
            clahe_clip_limit,
            clahe_grid_size,
        )

    def apply(
        self,
        img: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply grayscale conversion to a single image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image with shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns
        -------
        numpy.ndarray
            Grayscale-replicated image with shape ``(H, W, 3)`` and
            dtype ``uint8``.
        """
        # Convert to single-channel grayscale
        gray = self._handler.to_grayscale(img)

        # Enhance contrast (matches inference pipeline)
        if self._enhance:
            gray = self._handler.enhance_grayscale(gray)

        # Replicate to 3 channels — model expects (H, W, 3)
        replicated = np.stack([gray, gray, gray], axis=2)

        return replicated

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Return names of ``__init__`` args for serialisation.

        Required by albumentations for ``to_dict()`` / ``from_dict()``
        round-tripping.
        """
        return (
            "enhance",
            "clahe_clip_limit",
            "clahe_grid_size",
        )



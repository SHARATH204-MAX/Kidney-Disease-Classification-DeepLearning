"""
Single source of truth for the model input contract.

Every producer of model input tensors -- the training generator, the evaluation
generator and the Flask inference service -- must build them *identically*.
When these three call sites drift apart (raw 0-255 vs rescaled 0-1) the service
keeps running and keeps answering, just with garbage: that is exactly how this
project ended up predicting "Normal" for every scan.

The required scaling is backbone-dependent:

* ``EfficientNetB0`` starts with a ``Rescaling(1/255)`` layer inside the
  model graph, so callers must feed **raw float pixels in [0, 255]**. Dividing
  by 255 here would make the model see ~[0, 0.004] and destroy all signal.
* ``VGG16`` has no internal rescaling, so callers must divide by 255 first
  (and must NOT use ``preprocess_input`` unless training does too).

Keep the mapping below in sync with ``BACKBONES`` in
``components/prepair_base_model.py``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Geometry contract (VGG16 / EfficientNetB0 native resolution).
TARGET_SIZE = (224, 224)
INTERPOLATION = "bilinear"

# Must equal the training split, otherwise evaluation either leaks images the
# model has seen or skips a slice of the data.
VALIDATION_SPLIT = 0.20

# Where the caller feeds pixels the backbone rescales internally.
INTERNAL_RESCALING = "internal"
# Where the caller must normalise pixels to [0, 1] before feeding the model.
UNIT_RANGE = "unit"

DEFAULT_BACKBONE = "EfficientNetB0"

# backbone name -> scaling mode. Add a backbone here whenever you add it to
# components/prepair_base_model.py.
BACKBONE_SCALING = {
    "EfficientNetB0": INTERNAL_RESCALING,  # first layer is Rescaling(1/255)
    "VGG16": UNIT_RANGE,                   # expects inputs in [0, 1]
}


def scaling_mode(backbone: str = DEFAULT_BACKBONE) -> str:
    """Return the scaling mode for ``backbone`` (fails loudly on typos)."""
    try:
        return BACKBONE_SCALING[backbone]
    except KeyError:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Known backbones: "
            f"{sorted(BACKBONE_SCALING)}"
        ) from None


def generator_kwargs(
    backbone: str = DEFAULT_BACKBONE,
    validation_split: float = VALIDATION_SPLIT,
) -> dict:
    """Keyword args shared by every ``ImageDataGenerator`` construction."""
    return {
        # 1/255 for UNIT_RANGE backbones, None for INTERNAL_RESCALING ones.
        "rescale": 1.0 / 255 if scaling_mode(backbone) == UNIT_RANGE else None,
        "validation_split": validation_split,
    }


def dataflow_kwargs(batch_size: int) -> dict:
    """Keyword args shared by every ``flow_from_directory`` call."""
    return {
        "target_size": TARGET_SIZE,
        "batch_size": batch_size,
        "interpolation": INTERPOLATION,
    }


def prepare_image(img: Image.Image, backbone: str = DEFAULT_BACKBONE) -> np.ndarray:
    """Convert a PIL image into a single model-ready batch ``(1, H, W, 3)``."""
    img = img.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32)
    if scaling_mode(backbone) == UNIT_RANGE:
        array = array / 255.0
    return np.expand_dims(array, axis=0)

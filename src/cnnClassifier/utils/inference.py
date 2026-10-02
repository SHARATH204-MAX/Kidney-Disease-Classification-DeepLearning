"""How model input tensors become class probabilities.

Single source of truth for *inference-time probability computation*. Evaluation
(``scores.json``), the Flask service and any offline analysis must call this
function, otherwise the number published in ``scores.json`` does not describe
the model that ships — the same class of drift as the old train/serve rescale
mismatch (see ``utils/preprocessing.py``).

Test-time augmentation (TTA): average the softmax of the image with the softmax
of its horizontal flip. Left/right mirroring is anatomically valid for kidney
CT slices, and averaging cancels orientation-specific over-confidence. Measured
on the validation split: 0.8858 -> 0.9096 accuracy at threshold 0.5.

Set ``tta=False`` only for diagnostics (e.g. to reproduce a pre-TTA baseline).
"""

from __future__ import annotations

import numpy as np


def predict_probabilities(model, images: np.ndarray, tta: bool = True) -> np.ndarray:
    """Return ``(N, num_classes)`` softmax probabilities for a batch.

    ``images`` must already satisfy the backbone input contract
    (``utils/preprocessing.py``): raw 0-255 for EfficientNetB0, [0, 1] for VGG16.
    """
    images = np.asarray(images)
    p = np.asarray(model(images, training=False).numpy())

    if tta:
        # ascontiguousarray: TF refuses numpy views with negative strides.
        flipped = np.ascontiguousarray(images[:, :, ::-1, :])
        p_flip = np.asarray(model(flipped, training=False).numpy())
        p = 0.5 * (p + p_flip)

    return p

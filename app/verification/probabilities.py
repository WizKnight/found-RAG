import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise ValueError(
            "softmax expects a one-dimensional array."
        )

    shifted = logits - np.max(logits)

    exponentials = np.exp(shifted)

    return exponentials / np.sum(exponentials)
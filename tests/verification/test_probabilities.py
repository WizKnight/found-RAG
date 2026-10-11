import numpy as np
import pytest

from app.verification.probabilities import softmax


def test_softmax_probabilities_sum_to_one() -> None:
    logits = np.array(
        [-5.38, 4.55, -0.51]
    )

    probabilities = softmax(logits)

    assert probabilities.shape == (3,)
    assert np.sum(probabilities) == pytest.approx(1.0)


def test_softmax_preserves_ordering() -> None:
    logits = np.array(
        [1.0, 3.0, 2.0]
    )

    probabilities = softmax(logits)

    assert probabilities[1] > probabilities[2]
    assert probabilities[2] > probabilities[0]


def test_softmax_rejects_non_1d_input() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        softmax(logits)
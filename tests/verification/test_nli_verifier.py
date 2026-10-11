from uuid import uuid4

import numpy as np
import pytest

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    VerificationStatus,
)
from app.verification.nli_verifier import NLIClaimVerifier


class FakeNLIModel:
    def __init__(self, predictions: list[list[float]]) -> None:
        self.predictions = predictions

    def predict(
        self,
        pairs: list[list[str]],
        show_progress_bar: bool = False,
    ):
        return np.array(self.predictions)


def make_claim() -> Claim:
    return Claim(
        claim_id="CLAIM-1",
        text="Employees can carry unused leave into the next year.",
        citation_ids=["CIT-1"],
    )


def make_evidence() -> list[Citation]:
    return [
        Citation(
            citation_id="CIT-1",
            chunk_id=str(uuid4()),
            text=(
                "Unused annual leave may be carried forward "
                "to the following calendar year."
            ),
        )
    ]


def test_nli_entailment_is_supported() -> None:
    verifier = NLIClaimVerifier.__new__(
        NLIClaimVerifier
    )

    setattr(
        verifier,
        "model",
        FakeNLIModel(
            [
                [-5.38, 4.55, -0.50]
            ]
        ),
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    assert result.status == VerificationStatus.SUPPORTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.entailment_probability is not None
    assert result.entailment_probability > 0.5


def test_nli_contradiction_is_contradicted() -> None:
    verifier = NLIClaimVerifier.__new__(
        NLIClaimVerifier
    )

    setattr(
        verifier,
        "model",
        FakeNLIModel(
            [
                [3.88, -2.32, -1.37]
            ]
        ),
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    assert result.status == VerificationStatus.CONTRADICTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.contradiction_probability is not None
    assert result.contradiction_probability > 0.5


def test_nli_neutral_is_not_supported() -> None:
    verifier = NLIClaimVerifier.__new__(
        NLIClaimVerifier
    )

    setattr(
        verifier,
        "model",
        FakeNLIModel(
            [
                [-1.0, -2.0, 4.0]
            ]
        ),
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.neutral_probability is not None
    assert result.neutral_probability > 0.5



def test_nli_missing_evidence_is_not_supported() -> None:
    verifier = NLIClaimVerifier.__new__(
        NLIClaimVerifier
    )

    setattr(
        verifier,
        "model",
        FakeNLIModel(
            [
                [-5.38, 4.55, -0.50]
            ]
        ),
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=[],
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert result.confidence == 0.0
    assert result.evidence == []


def test_nli_probabilities_sum_to_one() -> None:
    verifier = NLIClaimVerifier.__new__(
        NLIClaimVerifier
    )

    setattr(
        verifier,
        "model",
        FakeNLIModel(
            [
                [-5.38, 4.55, -0.50]
            ]
        ),
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    entailment_probability = result.entailment_probability
    contradiction_probability = result.contradiction_probability
    neutral_probability = result.neutral_probability

    assert entailment_probability is not None
    assert contradiction_probability is not None
    assert neutral_probability is not None

    total = (
        entailment_probability
        + contradiction_probability
        + neutral_probability
    )

    assert total == pytest.approx(1.0)
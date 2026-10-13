
from typing import cast
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
        self.received_pairs: list[list[str]] = []

    def predict(
        self,
        pairs: list[list[str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.received_pairs = pairs
        return np.asarray(self.predictions)


def make_claim(citation_count: int = 1) -> Claim:
    return Claim(
        claim_id="CLAIM-1",
        text="Employees can carry unused leave into the next year.",
        citation_ids=[
            f"CIT-{index}"
            for index in range(1, citation_count + 1)
        ],
    )


def make_evidence(count: int = 1) -> list[Citation]:
    return [
        Citation(
            citation_id=f"CIT-{index}",
            chunk_id=str(uuid4()),
            text=(
                f"Evidence {index}: unused annual leave "
                "may be carried forward to the following year."
            ),
        )
        for index in range(1, count + 1)
    ]


def make_verifier(
    predictions: list[list[float]],
) -> NLIClaimVerifier:
    verifier = NLIClaimVerifier.__new__(NLIClaimVerifier)
    verifier.model = FakeNLIModel(predictions)
    return verifier


def test_nli_entailment_is_supported() -> None:
    verifier = make_verifier([[-5.38, 4.55, -0.50]])

    result = verifier.verify(make_claim(), make_evidence())

    assert result.status == VerificationStatus.SUPPORTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.entailment_probability is not None
    assert result.entailment_probability > 0.5
    assert len(result.evidence_verifications) == 1
    assert (
        result.evidence_verifications[0].status
        == VerificationStatus.SUPPORTED
    )


def test_nli_contradiction_is_contradicted() -> None:
    verifier = make_verifier([[3.88, -2.32, -1.37]])

    result = verifier.verify(make_claim(), make_evidence())

    assert result.status == VerificationStatus.CONTRADICTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.contradiction_probability is not None
    assert result.contradiction_probability > 0.5


def test_nli_neutral_is_not_supported() -> None:
    verifier = make_verifier([[-1.0, -2.0, 4.0]])

    result = verifier.verify(make_claim(), make_evidence())

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.neutral_probability is not None
    assert result.neutral_probability > 0.5


def test_nli_missing_evidence_is_not_supported() -> None:
    verifier = make_verifier([[-5.38, 4.55, -0.50]])

    result = verifier.verify(make_claim(), [])

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert result.confidence == 0.0
    assert result.evidence == []
    assert result.evidence_verifications == []
    assert result.neutral_probability == 1.0


def test_nli_probabilities_sum_to_one_per_citation() -> None:
    verifier = make_verifier([[-5.38, 4.55, -0.50]])

    result = verifier.verify(make_claim(), make_evidence())

    evidence_result = result.evidence_verifications[0]
    probabilities = [
        evidence_result.entailment_probability,
        evidence_result.contradiction_probability,
        evidence_result.neutral_probability,
    ]

    assert all(probability is not None for probability in probabilities)
    non_none_probabilities: list[float] = [
        probability
        for probability in probabilities
        if probability is not None
    ]
    assert sum(non_none_probabilities) == pytest.approx(1.0)


def test_multiple_supporting_citations_are_supported() -> None:
    verifier = make_verifier([
        [-5.0, 4.0, -1.0],
        [-4.0, 3.0, -1.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.SUPPORTED
    assert result.confidence is not None
    assert len(result.evidence_verifications) == 2
    assert all(
        item.status == VerificationStatus.SUPPORTED
        for item in result.evidence_verifications
    )


def test_multiple_contradictory_citations_are_contradicted() -> None:
    verifier = make_verifier([
        [4.0, -2.0, -1.0],
        [3.0, -2.0, -1.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.CONTRADICTED
    assert all(
        item.status == VerificationStatus.CONTRADICTED
        for item in result.evidence_verifications
    )


def test_support_and_contradiction_are_conflicting() -> None:
    verifier = make_verifier([
        [-5.0, 4.0, -1.0],
        [4.0, -2.0, -1.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.CONFLICTING
    assert result.confidence is None
    assert (
        result.evidence_verifications[0].status
        == VerificationStatus.SUPPORTED
    )
    assert (
        result.evidence_verifications[1].status
        == VerificationStatus.CONTRADICTED
    )


def test_neutral_citations_are_not_supported() -> None:
    verifier = make_verifier([
        [-1.0, -2.0, 4.0],
        [-2.0, -1.0, 3.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.NOT_SUPPORTED


def test_support_and_neutral_are_supported() -> None:
    verifier = make_verifier([
        [-5.0, 4.0, -1.0],
        [-1.0, -2.0, 4.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.SUPPORTED


def test_contradiction_and_neutral_are_contradicted() -> None:
    verifier = make_verifier([
        [4.0, -2.0, -1.0],
        [-1.0, -2.0, 4.0],
    ])

    result = verifier.verify(make_claim(), make_evidence(2))

    assert result.status == VerificationStatus.CONTRADICTED


def test_nli_creates_one_prediction_pair_per_citation() -> None:
    verifier = make_verifier([
        [-5.0, 4.0, -1.0],
        [4.0, -2.0, -1.0],
    ])
    evidence = make_evidence(2)

    verifier.verify(make_claim(), evidence)

    model = cast(FakeNLIModel, verifier.model)

    assert len(model.received_pairs) == 2
    assert model.received_pairs[0] == [
        evidence[0].text,
        make_claim().text,
    ]
    assert model.received_pairs[1] == [
        evidence[1].text,
        make_claim().text,
    ]


def test_nli_rejects_mismatched_prediction_count() -> None:
    verifier = make_verifier([[-5.0, 4.0, -1.0]])

    with pytest.raises(ValueError, match="Expected NLI model output"):
        verifier.verify(make_claim(), make_evidence(2))


def test_nli_rejects_invalid_prediction_shape() -> None:
    verifier = make_verifier([
        [-5.0, 4.0],
    ])

    with pytest.raises(ValueError, match="Expected NLI model output"):
        verifier.verify(make_claim(), make_evidence())

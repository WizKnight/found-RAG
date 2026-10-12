
from uuid import uuid4

import pytest

from app.models.generation import Citation, GeneratedAnswer
from app.models.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)
from app.verification.metrics import calculate_verification_metrics


def make_citation(citation_id: str) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=str(uuid4()),
        text=f"Evidence for {citation_id}.",
    )


def make_verification(
    claim_id: str,
    status: VerificationStatus,
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        claim=f"Claim {claim_id}",
        status=status,
    )


def test_calculates_claim_and_citation_metrics() -> None:
    answer = GeneratedAnswer(
        answer=(
            "Claim one is supported. [CIT-1] "
            "Claim two is unsupported. [CIT-2] "
            "Claim three is contradicted. [CIT-3] "
            "Claim four has conflicting evidence. [CIT-404]"
        ),
        citations=[
            make_citation("CIT-1"),
            make_citation("CIT-2"),
            make_citation("CIT-3"),
        ],
    )

    verification_result = VerificationResult(
        claim_verifications=[
            make_verification(
                "CLAIM-1",
                VerificationStatus.SUPPORTED,
            ),
            make_verification(
                "CLAIM-2",
                VerificationStatus.NOT_SUPPORTED,
            ),
            make_verification(
                "CLAIM-3",
                VerificationStatus.CONTRADICTED,
            ),
            make_verification(
                "CLAIM-4",
                VerificationStatus.CONFLICTING,
            ),
        ],
    )

    metrics = calculate_verification_metrics(
        generated_answer=answer,
        verification_result=verification_result,
        verification_latency_ms=125.5,
    )

    assert metrics.total_claims == 4
    assert metrics.supported_claims == 1
    assert metrics.not_supported_claims == 1
    assert metrics.contradicted_claims == 1
    assert metrics.conflicting_claims == 1
    assert metrics.unsupported_claims == 3

    assert metrics.supported_claim_rate == pytest.approx(0.25)
    assert metrics.not_supported_claim_rate == pytest.approx(0.25)
    assert metrics.contradiction_rate == pytest.approx(0.25)
    assert metrics.conflict_rate == pytest.approx(0.25)
    assert metrics.unsupported_claim_rate == pytest.approx(0.75)

    assert metrics.total_citation_references == 4
    assert metrics.invalid_citation_references == 1
    assert metrics.invalid_citation_rate == pytest.approx(0.25)
    assert metrics.verification_latency_ms == pytest.approx(125.5)


def test_empty_results_return_zero_rates() -> None:
    answer = GeneratedAnswer(answer="", citations=[])
    verification_result = VerificationResult(
        claim_verifications=[]
    )

    metrics = calculate_verification_metrics(
        generated_answer=answer,
        verification_result=verification_result,
    )

    assert metrics.total_claims == 0
    assert metrics.unsupported_claims == 0
    assert metrics.supported_claim_rate == 0.0
    assert metrics.unsupported_claim_rate == 0.0
    assert metrics.invalid_citation_rate == 0.0
    assert metrics.total_citation_references == 0
    assert metrics.invalid_citation_references == 0
    assert metrics.verification_latency_ms is None


def test_repeated_invalid_citation_references_are_counted() -> None:
    answer = GeneratedAnswer(
        answer=(
            "First claim is supported. [CIT-999] "
            "Second claim is also supported. [CIT-999]"
        ),
        citations=[],
    )

    verification_result = VerificationResult(
        claim_verifications=[
            make_verification(
                "CLAIM-1",
                VerificationStatus.NOT_SUPPORTED,
            ),
            make_verification(
                "CLAIM-2",
                VerificationStatus.NOT_SUPPORTED,
            ),
        ],
    )

    metrics = calculate_verification_metrics(
        generated_answer=answer,
        verification_result=verification_result,
    )

    assert metrics.total_citation_references == 2
    assert metrics.invalid_citation_references == 2
    assert metrics.invalid_citation_rate == pytest.approx(1.0)

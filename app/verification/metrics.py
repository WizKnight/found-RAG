
from app.models.generation import GeneratedAnswer
from app.models.verification import (
    VerificationResult,
    VerificationStatus,
)
from app.models.verification_metrics import VerificationMetrics
from app.verification.claim_extractor import extract_claims


def calculate_verification_metrics(
    generated_answer: GeneratedAnswer,
    verification_result: VerificationResult,
    verification_latency_ms: float | None = None,
) -> VerificationMetrics:
    verifications = verification_result.claim_verifications

    supported = sum(
        item.status == VerificationStatus.SUPPORTED
        for item in verifications
    )
    not_supported = sum(
        item.status == VerificationStatus.NOT_SUPPORTED
        for item in verifications
    )
    contradicted = sum(
        item.status == VerificationStatus.CONTRADICTED
        for item in verifications
    )
    conflicting = sum(
        item.status == VerificationStatus.CONFLICTING
        for item in verifications
    )

    total_claims = len(verifications)
    unsupported = total_claims - supported

    def rate(count: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return count / denominator

    # Compare every citation reference in the answer with the
    # authoritative citation IDs attached to GeneratedAnswer.
    claims = extract_claims(generated_answer.answer)
    known_citation_ids = {
        citation.citation_id
        for citation in generated_answer.citations
    }

    citation_references = [
        citation_id
        for claim in claims
        for citation_id in claim.citation_ids
    ]

    invalid_citation_references = sum(
        citation_id not in known_citation_ids
        for citation_id in citation_references
    )

    total_citation_references = len(citation_references)

    return VerificationMetrics(
        total_claims=total_claims,
        supported_claims=supported,
        not_supported_claims=not_supported,
        contradicted_claims=contradicted,
        conflicting_claims=conflicting,
        unsupported_claims=unsupported,
        supported_claim_rate=rate(supported, total_claims),
        not_supported_claim_rate=rate(
            not_supported,
            total_claims,
        ),
        contradiction_rate=rate(contradicted, total_claims),
        conflict_rate=rate(conflicting, total_claims),
        unsupported_claim_rate=rate(unsupported, total_claims),
        total_citation_references=total_citation_references,
        invalid_citation_references=invalid_citation_references,
        invalid_citation_rate=rate(
            invalid_citation_references,
            total_citation_references,
        ),
        verification_latency_ms=verification_latency_ms,
    )

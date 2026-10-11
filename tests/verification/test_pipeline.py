from app.models.generation import Citation, GeneratedAnswer
from app.models.verification import VerificationStatus
from app.verification.lexical_verifier import LexicalClaimVerifier
from app.verification.pipeline import VerificationPipeline


def test_verification_pipeline() -> None:
    generated_answer = GeneratedAnswer(
        answer=(
            "Employees receive 24 days of paid annual leave. [CIT-1]"
        ),
        citations=[
            Citation(
                citation_id="CIT-1",
                chunk_id="chunk-1",
                document_id="document-1",
                page_number=1,
                text=(
                    "Employees are entitled to 24 days "
                    "of paid annual leave."
                ),
            )
        ],
    )

    pipeline = VerificationPipeline(
        verifier=LexicalClaimVerifier()
    )

    result = pipeline.verify(generated_answer)

    assert len(result.claim_verifications) == 1

    verification = result.claim_verifications[0]

    assert verification.claim_id == "CLAIM-1"
    assert verification.citation_ids == ["CIT-1"]
    assert verification.status == VerificationStatus.SUPPORTED

    assert result.all_supported is True
    assert result.supported_claim_rate == 1.0

def test_verification_pipeline_detects_unsupported_claim() -> None:
    generated_answer = GeneratedAnswer(
        answer=(
            "Employees receive a housing allowance. [CIT-1]"
        ),
        citations=[
            Citation(
                citation_id="CIT-1",
                chunk_id="chunk-1",
                document_id="document-1",
                page_number=1,
                text=(
                    "Employees are entitled to 24 days "
                    "of paid annual leave."
                ),
            )
        ],
    )

    pipeline = VerificationPipeline(
        verifier=LexicalClaimVerifier()
    )

    result = pipeline.verify(generated_answer)

    verification = result.claim_verifications[0]

    assert (
        verification.status
        == VerificationStatus.NOT_SUPPORTED
    )

    assert result.all_supported is False
    assert result.supported_claim_rate == 0.0


def test_verification_pipeline_handles_multiple_claims() -> None:
    generated_answer = GeneratedAnswer(
        answer=(
            "Employees receive 24 days of paid annual leave. [CIT-1] "
            "Employees must submit requests five working days "
            "in advance. [CIT-2]"
        ),
        citations=[
            Citation(
                citation_id="CIT-1",
                chunk_id="chunk-1",
                document_id="document-1",
                page_number=1,
                text=(
                    "Employees are entitled to 24 days "
                    "of paid annual leave."
                ),
            ),
            Citation(
                citation_id="CIT-2",
                chunk_id="chunk-2",
                document_id="document-1",
                page_number=1,
                text=(
                    "Employees should submit annual leave "
                    "requests at least five working days "
                    "before the intended leave period."
                ),
            ),
        ],
    )

    pipeline = VerificationPipeline(
        verifier=LexicalClaimVerifier()
    )

    result = pipeline.verify(generated_answer)

    assert len(result.claim_verifications) == 2

    assert all(
        verification.status == VerificationStatus.SUPPORTED
        for verification in result.claim_verifications
    )

    assert result.all_supported is True
    assert result.supported_claim_rate == 1.0


def test_invalid_citation_is_not_treated_as_evidence() -> None:
    generated_answer = GeneratedAnswer(
        answer=(
            "Employees receive 24 days of paid annual leave. [CIT-999]"
        ),
        citations=[
            Citation(
                citation_id="CIT-1",
                chunk_id="chunk-1",
                text=(
                    "Employees are entitled to 24 days "
                    "of paid annual leave."
                ),
            )
        ],
    )

    pipeline = VerificationPipeline(
        verifier=LexicalClaimVerifier()
    )

    result = pipeline.verify(generated_answer)

    verification = result.claim_verifications[0]

    assert verification.citation_ids == ["CIT-999"]

    assert (
        verification.status
        == VerificationStatus.NOT_SUPPORTED
    )

    assert verification.evidence == []

from uuid import uuid4

import pytest

from app.models.generation import Citation, GeneratedAnswer
from app.models.rag import RAGResponse
from app.models.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)
from app.services.rag_orchestrator import (
    GenerationPipeline,
    RAGOrchestrator,
    VerificationPipeline,
)


class FakeGenerationPipeline(GenerationPipeline):
    def __init__(self, answers: list[GeneratedAnswer]) -> None:
        self.answers = answers
        self.calls: list[dict[str, str | None]] = []

    def generate(
        self,
        query: str,
        verification_feedback: str | None = None,
    ) -> GeneratedAnswer:
        self.calls.append(
            {
                "query": query,
                "verification_feedback": verification_feedback,
            }
        )
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        return self.answers[index]


class FakeVerificationPipeline(VerificationPipeline):
    def __init__(
        self,
        results: list[VerificationResult],
    ) -> None:
        self.results = results
        self.calls = 0

    def verify(
        self,
        generated_answer: GeneratedAnswer,
    ) -> VerificationResult:
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


def make_answer(
    answer: str = "Employees receive 24 days of annual leave. [CIT-1]",
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer,
        citations=[
            Citation(
                citation_id="CIT-1",
                chunk_id=str(uuid4()),
                text="Employees are entitled to 24 days of paid annual leave.",
            )
        ],
    )


def make_verification(
    status: VerificationStatus,
) -> VerificationResult:
    return VerificationResult(
        claim_verifications=[
            ClaimVerification(
                claim_id="CLAIM-1",
                claim="Employees receive 24 days of annual leave.",
                citation_ids=["CIT-1"],
                status=status,
                explanation=f"Test status: {status.value}",
            )
        ]
    )


def make_orchestrator(
    answers: list[GeneratedAnswer],
    results: list[VerificationResult],
    max_attempts: int = 2,
) -> tuple[
    RAGOrchestrator,
    FakeGenerationPipeline,
    FakeVerificationPipeline,
]:
    generation = FakeGenerationPipeline(answers)
    verification = FakeVerificationPipeline(results)

    orchestrator = RAGOrchestrator(
        generation_pipeline=generation,
        verification_pipeline=verification,
        max_attempts=max_attempts,
    )

    return orchestrator, generation, verification


def test_returns_answer_when_all_claims_are_supported() -> None:
    answer = make_answer()
    orchestrator, generation, verification = make_orchestrator(
        answers=[answer],
        results=[make_verification(VerificationStatus.SUPPORTED)],
    )

    response = orchestrator.answer("How much annual leave do employees get?")

    assert isinstance(response, RAGResponse)
    assert response.answer == answer.answer
    assert response.citations == answer.citations
    assert response.attempts == 1
    assert response.abstained is False
    assert generation.calls[0]["verification_feedback"] is None
    assert verification.calls == 1
    assert response.metrics.verification_latency_ms is not None


def test_retries_then_returns_supported_answer() -> None:
    first_answer = make_answer(
        "Employees receive 30 days of annual leave. [CIT-1]"
    )
    corrected_answer = make_answer()

    orchestrator, generation, verification = make_orchestrator(
        answers=[first_answer, corrected_answer],
        results=[
            make_verification(VerificationStatus.CONTRADICTED),
            make_verification(VerificationStatus.SUPPORTED),
        ],
    )

    response = orchestrator.answer("How much annual leave do employees get?")

    assert response.answer == corrected_answer.answer
    assert response.attempts == 2
    assert response.abstained is False
    assert verification.calls == 2
    assert generation.calls[1]["verification_feedback"] is not None
    assert "CONTRADICTED" in generation.calls[1]["verification_feedback"]


@pytest.mark.parametrize(
    "status",
    [
        VerificationStatus.NOT_SUPPORTED,
        VerificationStatus.CONTRADICTED,
        VerificationStatus.CONFLICTING,
    ],
)
def test_abstains_when_verification_keeps_failing(
    status: VerificationStatus,
) -> None:
    orchestrator, generation, verification = make_orchestrator(
        answers=[make_answer(), make_answer()],
        results=[
            make_verification(status),
            make_verification(status),
        ],
        max_attempts=2,
    )

    response = orchestrator.answer("What is the leave policy?")

    assert response.abstained is True
    assert response.answer == orchestrator.abstention_message
    assert response.citations == []
    assert response.attempts == 2
    assert verification.calls == 2
    assert len(generation.calls) == 2


def test_does_not_accept_empty_verification_results() -> None:
    empty_result = VerificationResult(claim_verifications=[])

    orchestrator, _, verification = make_orchestrator(
        answers=[make_answer(), make_answer()],
        results=[empty_result, empty_result],
    )

    response = orchestrator.answer("What is the leave policy?")

    assert response.abstained is True
    assert response.attempts == 2
    assert verification.calls == 2


def test_rejects_invalid_retry_budget() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RAGOrchestrator(
            generation_pipeline=FakeGenerationPipeline([make_answer()]),
            verification_pipeline=FakeVerificationPipeline(
                [make_verification(VerificationStatus.SUPPORTED)]
            ),
            max_attempts=0,
        )

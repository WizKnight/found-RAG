
from time import perf_counter
from typing import Protocol

from app.generation.pipeline import GenerationPipeline
from app.models.generation import GeneratedAnswer
from app.models.rag import RAGResponse
from app.models.verification import (
    VerificationResult,
    VerificationStatus,
)
from app.models.verification_metrics import VerificationMetrics
from app.verification.metrics import calculate_verification_metrics
from app.verification.pipeline import VerificationPipeline


class SupportsGenerate(Protocol):
    def generate(
        self,
        *,
        query: str,
        verification_feedback: str | None = None,
    ) -> GeneratedAnswer:
        ...


class RAGOrchestrator:
    def __init__(
        self,
        generation_pipeline: GenerationPipeline,
        verification_pipeline: VerificationPipeline,
        max_attempts: int = 2,
        abstention_message: str = (
            "I cannot provide a reliable answer from the available evidence."
        ),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.generation_pipeline = generation_pipeline
        self.verification_pipeline = verification_pipeline
        self.max_attempts = max_attempts
        self.abstention_message = abstention_message

    def answer(self, query: str) -> RAGResponse:
        verification_feedback: str | None = None

        final_candidate: GeneratedAnswer | None = None
        final_verification: VerificationResult | None = None
        final_metrics: VerificationMetrics | None = None

        for attempt in range(1, self.max_attempts + 1):
            candidate = self.generation_pipeline.generate(
                query=query,
                verification_feedback=verification_feedback,
            )
            if candidate is None:
                raise RuntimeError(
                    "Generation pipeline returned no candidate for verification."
                )

            verification_start = perf_counter()
            verification = self.verification_pipeline.verify(candidate)
            verification_latency_ms = (
                perf_counter() - verification_start
            ) * 1000.0

            metrics = calculate_verification_metrics(
                generated_answer=candidate,
                verification_result=verification,
                verification_latency_ms=verification_latency_ms,
            )

            final_candidate = candidate
            final_verification = verification
            final_metrics = metrics

            if self._is_verified(verification):
                return RAGResponse(
                    answer=candidate.answer,
                    citations=candidate.citations,
                    verification=verification,
                    metrics=metrics,
                    attempts=attempt,
                    abstained=False,
                )

            if attempt < self.max_attempts:
                verification_feedback = self._build_feedback(
                    verification
                )

        # The loop always runs at least once because max_attempts >= 1.
        assert final_candidate is not None
        assert final_verification is not None
        assert final_metrics is not None

        return RAGResponse(
            answer=self.abstention_message,
            citations=[],
            verification=final_verification,
            metrics=final_metrics,
            attempts=self.max_attempts,
            abstained=True,
        )

    @staticmethod
    def _is_verified(
        verification: VerificationResult,
    ) -> bool:
        # An empty verification result must never count as a verified answer.
        return bool(verification.claim_verifications) and all(
            item.status == VerificationStatus.SUPPORTED
            for item in verification.claim_verifications
        )

    @staticmethod
    def _build_feedback(
        verification: VerificationResult,
    ) -> str:
        failed_claims = [
            item
            for item in verification.claim_verifications
            if item.status != VerificationStatus.SUPPORTED
        ]

        if not failed_claims:
            return (
                "The previous answer could not be verified. "
                "Provide only claims supported by the supplied evidence."
            )

        lines = [
            "The following claims failed verification. "
            "Correct or remove them using only the supplied evidence:"
        ]

        for item in failed_claims:
            explanation = item.explanation or (
                "The evidence did not establish this claim."
            )
            lines.append(
                f"- Claim: {item.claim}\n"
                f"  Status: {item.status.value.upper()}\n"
                f"  Reason: {explanation}"
            )

        return "\n".join(lines)


import numpy as np
from sentence_transformers import CrossEncoder

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
    EvidenceVerification,
    VerificationStatus,
)
from app.verification.probabilities import softmax
from app.verification.verifier import ClaimVerifier


class NLIClaimVerifier(ClaimVerifier):
    ENTAILMENT = 1
    CONTRADICTION = 0
    NEUTRAL = 2
    NUM_LABELS = 3

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)

    def verify(
        self,
        claim: Claim,
        evidence: list[Citation],
    ) -> ClaimVerification:
        if not evidence:
            return ClaimVerification(
                claim_id=claim.claim_id,
                claim=claim.text,
                citation_ids=claim.citation_ids,
                status=VerificationStatus.NOT_SUPPORTED,
                confidence=0.0,
                entailment_probability=0.0,
                contradiction_probability=0.0,
                neutral_probability=1.0,
                evidence=[],
                evidence_verifications=[],
                explanation="No cited evidence was found.",
            )

        pairs = [
            [citation.text, claim.text]
            for citation in evidence
        ]

        predictions = np.asarray(
            self.model.predict(
                pairs,
                show_progress_bar=False,
            )
        )

        expected_shape = (len(evidence), self.NUM_LABELS)
        if predictions.shape != expected_shape:
            raise ValueError(
                "Expected NLI model output with shape "
                f"{expected_shape}, got {predictions.shape}."
            )

        evidence_verifications: list[EvidenceVerification] = []

        for citation, prediction in zip(
            evidence,
            predictions,
            strict=True,
        ):
            probabilities = softmax(prediction)

            label_index = int(np.argmax(probabilities))
            confidence = float(probabilities[label_index])
            status = self._label_to_status(label_index)

            evidence_verifications.append(
                EvidenceVerification(
                    citation_id=citation.citation_id,
                    status=status,
                    confidence=confidence,
                    entailment_probability=float(
                        probabilities[self.ENTAILMENT]
                    ),
                    contradiction_probability=float(
                        probabilities[self.CONTRADICTION]
                    ),
                    neutral_probability=float(
                        probabilities[self.NEUTRAL]
                    ),
                    evidence=citation.text,
                )
            )

        status = self._aggregate_status(evidence_verifications)

        entailment_probability = max(
            result.entailment_probability
            for result in evidence_verifications
            if result.entailment_probability is not None
        )
        contradiction_probability = max(
            result.contradiction_probability
            for result in evidence_verifications
            if result.contradiction_probability is not None
        )
        neutral_probability = max(
            result.neutral_probability
            for result in evidence_verifications
            if result.neutral_probability is not None
        )

        # Conflicting evidence has no single, representative confidence.
        confidence = (
            None
            if status == VerificationStatus.CONFLICTING
            else max(
                result.confidence
                for result in evidence_verifications
                if result.confidence is not None
            )
        )

        return ClaimVerification(
            claim_id=claim.claim_id,
            claim=claim.text,
            citation_ids=claim.citation_ids,
            status=status,
            confidence=confidence,
            entailment_probability=entailment_probability,
            contradiction_probability=contradiction_probability,
            neutral_probability=neutral_probability,
            evidence=[
                result.evidence
                for result in evidence_verifications
            ],
            evidence_verifications=evidence_verifications,
            explanation=self._build_explanation(status),
        )

    @staticmethod
    def _aggregate_status(
        results: list[EvidenceVerification],
    ) -> VerificationStatus:
        statuses = {result.status for result in results}

        has_support = VerificationStatus.SUPPORTED in statuses
        has_contradiction = VerificationStatus.CONTRADICTED in statuses

        if has_support and has_contradiction:
            return VerificationStatus.CONFLICTING

        if has_contradiction:
            return VerificationStatus.CONTRADICTED

        if has_support:
            return VerificationStatus.SUPPORTED

        return VerificationStatus.NOT_SUPPORTED

    @staticmethod
    def _build_explanation(
        status: VerificationStatus,
    ) -> str:
        explanations = {
            VerificationStatus.SUPPORTED: (
                "At least one cited source supports the claim."
            ),
            VerificationStatus.CONTRADICTED: (
                "The cited evidence contradicts the claim."
            ),
            VerificationStatus.NOT_SUPPORTED: (
                "The cited evidence does not support the claim."
            ),
            VerificationStatus.CONFLICTING: (
                "The cited evidence contains both supporting "
                "and contradicting statements."
            ),
        }
        return explanations[status]

    @classmethod
    def _label_to_status(
        cls,
        label_index: int,
    ) -> VerificationStatus:
        if label_index == cls.ENTAILMENT:
            return VerificationStatus.SUPPORTED

        if label_index == cls.CONTRADICTION:
            return VerificationStatus.CONTRADICTED

        if label_index == cls.NEUTRAL:
            return VerificationStatus.NOT_SUPPORTED

        raise ValueError(
            f"Unknown NLI label index: {label_index}"
        )

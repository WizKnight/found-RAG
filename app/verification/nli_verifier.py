import numpy as np
from sentence_transformers import CrossEncoder

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
    VerificationStatus,
)
from app.verification.probabilities import softmax
from app.verification.verifier import ClaimVerifier


class NLIClaimVerifier(ClaimVerifier):
    ENTAILMENT = 1
    CONTRADICTION = 0
    NEUTRAL = 2

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

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
                explanation="No cited evidence was found.",
            )

        pairs = [
            [citation.text, claim.text]
            for citation in evidence
        ]

        predictions = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        predictions = np.asarray(predictions)

        if predictions.ndim != 2:
            raise ValueError(
                "Expected NLI model output with shape "
                "(num_pairs, num_labels)."
            )

        best_result: tuple[
            VerificationStatus,
            float,
            np.ndarray,
        ] | None = None

        for prediction in predictions:
            probabilities = softmax(prediction)

            label_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[label_index]
            )

            status = self._label_to_status(
                label_index
            )

            if (
                best_result is None
                or confidence > best_result[1]
            ):
                best_result = (status, confidence, probabilities)

        assert best_result is not None

        status, confidence, probabilities = best_result

        return ClaimVerification(
            claim_id=claim.claim_id,
            claim=claim.text,
            citation_ids=claim.citation_ids,
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
            evidence=[
                citation.text
                for citation in evidence
            ],
            explanation=(
                "NLI verification using "
                "cross-encoder/nli-deberta-v3-base."
            ),
        )

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
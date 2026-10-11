import re

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
    VerificationStatus,
)
from app.verification.verifier import ClaimVerifier


class LexicalClaimVerifier(ClaimVerifier):
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
                evidence=[],
                explanation="No cited evidence was found.",
            )

        claim_tokens = self._tokens(claim.text)

        evidence_text = " ".join(
            citation.text
            for citation in evidence
        )

        evidence_tokens = self._tokens(evidence_text)

        if not claim_tokens:
            status = VerificationStatus.NOT_SUPPORTED
        else:
            overlap = (
                len(claim_tokens & evidence_tokens)
                / len(claim_tokens)
            )

            status = (
                VerificationStatus.SUPPORTED
                if overlap >= 0.5
                else VerificationStatus.NOT_SUPPORTED
            )

        return ClaimVerification(
            claim_id=claim.claim_id,
            claim=claim.text,
            citation_ids=claim.citation_ids,
            status=status,
            evidence=[
                citation.text
                for citation in evidence
            ],
            explanation="Lexical evidence overlap baseline.",
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )
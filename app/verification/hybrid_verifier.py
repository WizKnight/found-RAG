from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
    VerificationStatus,
)
from app.verification.lexical_verifier import LexicalClaimVerifier
from app.verification.nli_verifier import NLIClaimVerifier
from app.verification.verifier import ClaimVerifier


class HybridClaimVerifier(ClaimVerifier):
    def __init__(
        self,
        lexical_verifier: LexicalClaimVerifier,
        nli_verifier: NLIClaimVerifier,
        lexical_threshold: float = 0.5,
    ) -> None:
        self.lexical_verifier = lexical_verifier
        self.nli_verifier = nli_verifier
        self.lexical_threshold = lexical_threshold

    def verify(
        self,
        claim: Claim,
        evidence: list[Citation],
    ) -> ClaimVerification:
        if not evidence:
            return self.nli_verifier.verify(
                claim=claim,
                evidence=evidence,
            )

        lexical_result = self.lexical_verifier.verify(
            claim=claim,
            evidence=evidence,
        )

        if lexical_result.status == VerificationStatus.NOT_SUPPORTED:
            return lexical_result

        return self.nli_verifier.verify(
            claim=claim,
            evidence=evidence,
        )
from app.models.generation import GeneratedAnswer
from app.models.verification import VerificationResult
from app.verification.claim_extractor import extract_claims
from app.verification.evidence import EvidenceResolver
from app.verification.verifier import ClaimVerifier


class VerificationPipeline:
    def __init__(
        self,
        verifier: ClaimVerifier,
    ) -> None:
        self.verifier = verifier

    def verify(
        self,
        generated_answer: GeneratedAnswer,
    ) -> VerificationResult:
        claims = extract_claims(
            generated_answer.answer
        )

        evidence_resolver = EvidenceResolver(
            generated_answer.citations
        )

        verifications = []

        for claim in claims:
            evidence = evidence_resolver.resolve(
                claim.citation_ids
            )

            verification = self.verifier.verify(
                claim=claim,
                evidence=evidence,
            )

            verifications.append(verification)

        return VerificationResult(
            claim_verifications=verifications
        )
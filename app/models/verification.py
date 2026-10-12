from enum import StrEnum

from pydantic import BaseModel, Field


class VerificationStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_SUPPORTED = "not_supported"
    CONFLICTING = "conflicting"


class Claim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class EvidenceVerification(BaseModel):
    citation_id: str
    status: VerificationStatus
    confidence: float | None = None
    entailment_probability: float | None = None
    contradiction_probability: float | None = None
    neutral_probability: float | None = None
    evidence: str


class ClaimVerification(BaseModel):
    claim_id: str
    claim: str
    citation_ids: list[str] = Field(default_factory=list)

    status: VerificationStatus

    confidence: float | None = None

    entailment_probability: float | None = None
    contradiction_probability: float | None = None
    neutral_probability: float | None = None

    evidence: list[str] = Field(default_factory=list)

    evidence_verifications: list[EvidenceVerification] = Field(
        default_factory=list
    )

    explanation: str | None = None


class VerificationResult(BaseModel):
    claim_verifications: list[ClaimVerification] = Field(
        default_factory=list
    )

    @property
    def all_supported(self) -> bool:
        return all(
            verification.status == VerificationStatus.SUPPORTED
            for verification in self.claim_verifications
        )

    @property
    def supported_claim_rate(self) -> float:
        if not self.claim_verifications:
            return 0.0

        supported = sum(
            verification.status == VerificationStatus.SUPPORTED
            for verification in self.claim_verifications
        )

        return supported / len(self.claim_verifications)
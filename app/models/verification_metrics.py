
from pydantic import BaseModel, Field


class VerificationMetrics(BaseModel):
    total_claims: int = Field(ge=0)

    supported_claims: int = Field(ge=0)
    not_supported_claims: int = Field(ge=0)
    contradicted_claims: int = Field(ge=0)
    conflicting_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)

    supported_claim_rate: float = Field(ge=0.0, le=1.0)
    not_supported_claim_rate: float = Field(ge=0.0, le=1.0)
    contradiction_rate: float = Field(ge=0.0, le=1.0)
    conflict_rate: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)

    total_citation_references: int = Field(ge=0)
    invalid_citation_references: int = Field(ge=0)
    invalid_citation_rate: float = Field(ge=0.0, le=1.0)

    verification_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

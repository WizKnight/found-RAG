
from pydantic import BaseModel, Field

from app.models.generation import Citation
from app.models.verification import VerificationResult
from app.models.verification_metrics import VerificationMetrics


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    verification: VerificationResult
    metrics: VerificationMetrics
    attempts: int = Field(ge=1)
    abstained: bool = False

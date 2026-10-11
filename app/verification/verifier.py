from abc import ABC, abstractmethod

from app.models.verification import (
    Claim,
    ClaimVerification,
)
from app.models.generation import Citation


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        claim: Claim,
        evidence: list[Citation],
    ) -> ClaimVerification:
        ...
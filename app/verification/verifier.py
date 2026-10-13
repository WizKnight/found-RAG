from abc import ABC, abstractmethod

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
)


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        claim: Claim,
        evidence: list[Citation],
    ) -> ClaimVerification:
        ...
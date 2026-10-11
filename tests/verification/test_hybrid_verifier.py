 # type: ignore

from uuid import uuid4

from app.models.generation import Citation
from app.models.verification import (
    Claim,
    ClaimVerification,
    VerificationStatus,
)
from app.verification.hybrid_verifier import HybridClaimVerifier


class FakeLexicalVerifier:
    def __init__(self, result: ClaimVerification) -> None:
        self.result = result
        self.calls = 0

    def verify(self, claim, evidence):
        self.calls += 1
        return self.result


class FakeNLIVerifier:
    def __init__(self, result: ClaimVerification) -> None:
        self.result = result
        self.calls = 0

    def verify(self, claim, evidence):
        self.calls += 1
        return self.result


def make_claim() -> Claim:
    return Claim(
        claim_id="CLAIM-1",
        text="Employees receive 24 days of annual leave.",
        citation_ids=["CIT-1"],
    )


def make_evidence() -> list[Citation]:
    return [
        Citation(
            citation_id="CIT-1",
            chunk_id=str(uuid4()),
            document_id=str(uuid4()),
            page_number=1,
            text="Employees are entitled to 24 days of paid annual leave.",
        )
    ]


def make_result(status: VerificationStatus) -> ClaimVerification:
    return ClaimVerification(
        claim_id="CLAIM-1",
        claim="Employees receive 24 days of annual leave.",
        citation_ids=["CIT-1"],
        status=status,
    )


def test_hybrid_short_circuits_when_lexical_verifier_rejects():
    lexical = FakeLexicalVerifier(
        make_result(VerificationStatus.NOT_SUPPORTED)
    )

    nli = FakeNLIVerifier(
        make_result(VerificationStatus.SUPPORTED)
    )

    verifier = HybridClaimVerifier(
        lexical_verifier=lexical,
        nli_verifier=nli,
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert lexical.calls == 1
    assert nli.calls == 0


def test_hybrid_uses_nli_when_lexical_verifier_passes():
    lexical = FakeLexicalVerifier(
        make_result(VerificationStatus.SUPPORTED)
    )

    nli = FakeNLIVerifier(
        make_result(VerificationStatus.SUPPORTED)
    )

    verifier = HybridClaimVerifier(
        lexical_verifier=lexical,
        nli_verifier=nli,
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=make_evidence(),
    )

    assert result.status == VerificationStatus.SUPPORTED
    assert lexical.calls == 1
    assert nli.calls == 1


def test_hybrid_uses_nli_for_missing_evidence():
    lexical = FakeLexicalVerifier(
        make_result(VerificationStatus.NOT_SUPPORTED)
    )

    nli = FakeNLIVerifier(
        make_result(VerificationStatus.NOT_SUPPORTED)
    )

    verifier = HybridClaimVerifier(
        lexical_verifier=lexical,
        nli_verifier=nli,
    )

    result = verifier.verify(
        claim=make_claim(),
        evidence=[],
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED
    assert lexical.calls == 0
    assert nli.calls == 1
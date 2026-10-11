import re

from app.models.verification import Claim
from app.verification.citation_parser import extract_citation_ids


CLAIM_PATTERN = re.compile(
    r"""
    .*?
    [.!?]
    (?:\s*\[CIT-\d+\])*
    """,
    re.VERBOSE | re.DOTALL,
)


def extract_claims(answer: str) -> list[Claim]:
    matches = CLAIM_PATTERN.findall(answer.strip())

    claims: list[Claim] = []

    for index, sentence in enumerate(matches, start=1):
        sentence = sentence.strip()

        if not sentence:
            continue

        citation_ids = extract_citation_ids(sentence)

        claim_text = re.sub(
            r"\s*\[CIT-\d+\]",
            "",
            sentence,
        ).strip()

        if not claim_text:
            continue

        claims.append(
            Claim(
                claim_id=f"CLAIM-{index}",
                text=claim_text,
                citation_ids=citation_ids,
            )
        )

    return claims
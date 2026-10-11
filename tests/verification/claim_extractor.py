import re

from app.models.verification import Claim
from app.verification.citation_parser import extract_citation_ids


def extract_claims(answer: str) -> list[Claim]:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    claims: list[Claim] = []

    for index, sentence in enumerate(
        sentences,
        start=1,
    ):
        sentence = sentence.strip()

        if not sentence:
            continue

        citation_ids = extract_citation_ids(sentence)

        claim_text = re.sub(
            r"\[CIT-\d+\]",
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
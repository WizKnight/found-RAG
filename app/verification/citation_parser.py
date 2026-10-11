import re

CITATION_PATTERN = re.compile(r"\[CIT-(\d+)\]")

def extract_citation_ids(text: str) -> list[str]:
    matches = CITATION_PATTERN.findall(text)

    return [
        f"CIT-{match}"
        for match in matches
    ]
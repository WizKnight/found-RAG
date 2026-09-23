import re


def normalize_text(text: str) -> str:
    """Normalize extracted document text while preserving meaning."""

    text = text.replace("\x00", " ")

    # Normalize whitespace while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
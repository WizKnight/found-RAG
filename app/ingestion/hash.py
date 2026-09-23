import hashlib
from pathlib import Path


def calculate_file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash for a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()
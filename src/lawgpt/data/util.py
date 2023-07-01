from hashlib import sha256
from pathlib import Path


def compute_sha256(filename: Path | str):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return sha256(f.read()).hexdigest()

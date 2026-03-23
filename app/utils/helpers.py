import hashlib
import time
from pathlib import Path
from typing import Any


def hash_text(text: str) -> str:
    """Return SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def hash_file(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories (including parents) if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def timer(func):
    """Decorator that logs the execution time of a function."""
    import functools
    from app.utils.logger import get_logger
    log = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result

    return wrapper


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text and append ellipsis if too long."""
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def flatten(nested: list[Any]) -> list[Any]:
    """Flatten one level of nesting in a list."""
    return [item for sublist in nested for item in sublist]
from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def normalize_title(title: str) -> str:
    collapsed = " ".join((title or "").strip().lower().split())
    no_marks = "".join(
        ch for ch in unicodedata.normalize("NFKD", collapsed) if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9 ]+", "", no_marks)


def normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    clean = doi.strip().lower()
    clean = re.sub(r"^https?://(dx\.)?doi\.org/", "", clean)
    return clean


def build_paper_id(doi: str, title: str) -> str:
    stable_key = normalize_doi(doi) or normalize_title(title)
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return f"paper_{digest[:16]}"


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    retries: int = 5,
    base_delay: float = 0.7,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error

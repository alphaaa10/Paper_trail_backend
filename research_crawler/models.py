from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PaperRecord:
    title: str
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    year: str = ""
    source: str = ""
    paper_url: str = ""
    pdf_url: str = ""
    paper_id: str = ""


@dataclass(slots=True)
class DownloadResult:
    paper_id: str
    status: str
    reason: str = ""
    pdf_path: Path | None = None
    metadata_path: Path | None = None


def sanitize_record(record: PaperRecord) -> PaperRecord:
    """Normalize all fields to safe, serializable values."""
    return PaperRecord(
        title=(record.title or "").strip(),
        authors=[a.strip() for a in (record.authors or []) if a and a.strip()],
        doi=(record.doi or "").strip(),
        year=str(record.year or "").strip(),
        source=(record.source or "").strip(),
        paper_url=(record.paper_url or "").strip(),
        pdf_url=(record.pdf_url or "").strip(),
        paper_id=(record.paper_id or "").strip(),
    )


def to_metadata_json(record: PaperRecord, pdf_path: Path) -> dict[str, Any]:
    return {
        "title": record.title,
        "authors": record.authors,
        "doi": record.doi,
        "year": record.year,
        "source": record.source,
        "pdf_url": record.pdf_url,
        "pdf_path": str(pdf_path.as_posix()),
    }

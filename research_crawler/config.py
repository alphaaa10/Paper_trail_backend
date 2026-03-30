from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    """Load variables from project .env if present."""
    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=root_dir / ".env", override=False)

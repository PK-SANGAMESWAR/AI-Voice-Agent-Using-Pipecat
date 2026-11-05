"""Shared filesystem and environment configuration for the local demo."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


def load_local_environment() -> None:
    """Load checked-in defaults first, then ignored local secrets."""
    load_dotenv(PROJECT_DIR / ".env", override=True)
    load_dotenv(PROJECT_DIR / ".env.local", override=True)


def configure_windows_console() -> None:
    """Keep Pipecat's Unicode runner output working in older Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


load_local_environment()
configure_windows_console()

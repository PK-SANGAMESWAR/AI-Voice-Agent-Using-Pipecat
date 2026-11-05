"""Simple local Markdown knowledge-base search for the voice agent."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from campusx_voice_agent.config import DATA_DIR

SUPPORTED_EXTENSIONS = {".md", ".txt"}
DEFAULT_KB_DIR = DATA_DIR / "knowledge_base"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KnowledgeMatch:
    source: str
    title: str
    snippet: str
    score: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
        }


def _knowledge_base_dir() -> Path:
    configured = os.getenv("CAMPUSX_KNOWLEDGE_BASE_DIR")
    return Path(configured) if configured else DEFAULT_KB_DIR


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _sections(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _compact(text: str, max_chars: int = 900) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def _score(query: str, query_tokens: set[str], text: str) -> int:
    lowered = text.lower()
    text_tokens = _tokens(text)
    overlap = len(query_tokens & text_tokens)
    phrase_bonus = 4 if query.strip().lower() in lowered else 0
    heading_bonus = 2 if text.lstrip().startswith("#") else 0
    return overlap + phrase_bonus + heading_bonus


def search_knowledge_base(
    query: str,
    *,
    limit: int = 4,
    kb_dir: Path | None = None,
) -> dict[str, object]:
    """Search local CampusX knowledge files and return short verified snippets."""
    cleaned_query = query.strip()
    root = kb_dir or _knowledge_base_dir()
    base_result: dict[str, object] = {
        "query": cleaned_query,
        "knowledge_base_dir": str(root),
        "current_date": date.today().isoformat(),
        "timezone": "Asia/Kolkata",
    }

    if not cleaned_query:
        return {
            **base_result,
            "found": False,
            "message": "No search query was provided.",
            "results": [],
        }

    if not root.exists():
        return {
            **base_result,
            "found": False,
            "message": "Knowledge base directory does not exist.",
            "results": [],
        }

    query_tokens = _tokens(cleaned_query)
    matches: list[KnowledgeMatch] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        title = _title_for(path, text)
        for section in _sections(text):
            score = _score(cleaned_query, query_tokens, section)
            if score > 0:
                matches.append(
                    KnowledgeMatch(
                        source=path.name,
                        title=title,
                        snippet=_compact(section),
                        score=score,
                    )
                )

    matches.sort(key=lambda match: (-match.score, match.source, match.snippet))
    limited = matches[: max(1, limit)]
    return {
        **base_result,
        "found": bool(limited),
        "message": (
            "Use these verified knowledge-base snippets when answering."
            if limited
            else "No verified knowledge-base result matched the query."
        ),
        "results": [match.as_dict() for match in limited],
    }
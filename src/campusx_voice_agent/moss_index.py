"""Create or refresh a Moss index from local CampusX knowledge-base files."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re

from loguru import logger

from campusx_voice_agent.config import DATA_DIR

KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
DEFAULT_INDEX_NAME = "campusx-voice-agent-knowledge"
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 250


def _require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing environment variable(s): {joined}. "
            "Set them in .env.local or your shell before creating a Moss index."
        )


def _import_moss_client_types():
    try:
        from inferedge_moss import DocumentInfo, MossClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing package: inferedge-moss. Run `uv sync --active` "
            "or `pip install pipecat-moss`."
        ) from exc
    return DocumentInfo, MossClient


def _normalize_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    text = _normalize_text(text)
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        boundary = text.rfind("\n## ", start, end)
        if boundary <= start + 400:
            boundary = text.rfind("\n\n", start, end)
        if boundary <= start + 400:
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        if boundary >= len(text):
            break
        start = max(0, boundary - CHUNK_OVERLAP)

    return chunks


def _index_info_matches_name(index_info, index_name: str) -> bool:
    """Return True when a Moss list_indexes() item represents index_name."""
    if isinstance(index_info, str):
        return index_info == index_name
    if isinstance(index_info, dict):
        candidates = (
            index_info.get("name"),
            index_info.get("indexName"),
            index_info.get("index_name"),
        )
    else:
        candidates = (
            getattr(index_info, "name", None),
            getattr(index_info, "indexName", None),
            getattr(index_info, "index_name", None),
        )
    return index_name in candidates


async def _moss_index_exists(client, index_name: str) -> bool:
    indexes = await client.list_indexes()
    return any(_index_info_matches_name(index_info, index_name) for index_info in indexes)


def _document_id(document) -> str | None:
    if isinstance(document, dict):
        return document.get("id")
    return getattr(document, "id", None)


async def _sync_existing_moss_index(client, index_name: str, documents) -> None:
    remote_documents = await client.get_docs(name=index_name)
    local_ids = {_document_id(document) for document in documents}
    remote_ids = {_document_id(document) for document in remote_documents}
    local_ids.discard(None)
    remote_ids.discard(None)

    stale_ids = sorted(remote_ids - local_ids)
    if stale_ids:
        logger.info(
            "Deleting {} stale chunks from Moss index '{}'",
            len(stale_ids),
            index_name,
        )
        await client.delete_docs(name=index_name, doc_ids=stale_ids)

    logger.info(
        "Adding or updating {} local chunks in Moss index '{}'",
        len(documents),
        index_name,
    )
    await client.add_docs(name=index_name, docs=documents)


def _load_documents():
    DocumentInfo, _ = _import_moss_client_types()
    markdown_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not markdown_files:
        raise RuntimeError(f"No markdown files found in {KNOWLEDGE_DIR}")

    documents = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(_chunk_text(text), start=1):
            digest = hashlib.sha1(f"{path.name}:{index}:{chunk}".encode()).hexdigest()[:12]
            documents.append(
                DocumentInfo(
                    id=f"{path.stem}-{index}-{digest}",
                    text=chunk,
                    metadata={
                        "source": path.name,
                        "chunk": str(index),
                        "project": "campusx-voice-agent",
                    },
                )
            )
    return documents


async def create_moss_index() -> None:
    """Upload local knowledge-base chunks to Moss using Moss APIs."""
    _require_env("MOSS_PROJECT_ID", "MOSS_PROJECT_KEY")
    _, MossClient = _import_moss_client_types()

    index_name = os.getenv("MOSS_INDEX_NAME", DEFAULT_INDEX_NAME)
    model_id = os.getenv("MOSS_MODEL_ID", "moss-minilm")
    client = MossClient(
        project_id=os.environ["MOSS_PROJECT_ID"],
        project_key=os.environ["MOSS_PROJECT_KEY"],
    )

    documents = _load_documents()
    if await _moss_index_exists(client, index_name):
        logger.info(
            "Moss index '{}' already exists; diff-syncing {} local chunks",
            index_name,
            len(documents),
        )
        await _sync_existing_moss_index(client, index_name, documents)
        logger.success("Moss index '{}' synced successfully", index_name)
    else:
        logger.info("Uploading {} chunks to Moss index '{}'", len(documents), index_name)
        await client.create_index(name=index_name, docs=documents, model_id=model_id)
        logger.success("Moss index '{}' created successfully", index_name)
    logger.info("Set MOSS_INDEX_NAME={} before running the voice agent", index_name)


def main() -> None:
    asyncio.run(create_moss_index())


if __name__ == "__main__":
    main()


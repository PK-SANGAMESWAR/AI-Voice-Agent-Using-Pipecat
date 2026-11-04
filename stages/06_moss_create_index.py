"""Stage 06: create a Moss index from the local CampusX knowledge base.

Run this before Stage 07 and whenever local knowledge-base files change. The
documents are uploaded to Moss through the Moss API; this stage does not create embeddings on your laptop.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Stage 06 is a standalone offline step, so paths are resolved from this file
# instead of from the live voice runner.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# The local markdown files are the source of truth for retrieval content.
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge_base"
# Use a stable default index name so Stage 07 can point to the same Moss index.
DEFAULT_INDEX_NAME = "campusx-voice-agent-knowledge"
# Chunk size balances retrieval quality with enough surrounding context for answers.
CHUNK_CHARS = 1800
# Overlap preserves context at chunk boundaries so important sentences are not split.
CHUNK_OVERLAP = 250


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing environment variable(s): {joined}. "
            "Set them in your local environment or ignored .env file."
        )


def import_moss_client_types():
    # Import Moss lazily so users can run earlier stages without installing
    # the optional retrieval dependency.
    try:
        from inferedge_moss import DocumentInfo, MossClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing package: inferedge-moss. Install it before this stage with "
            "`pip install pipecat-moss` or `uv add pipecat-moss`."
        ) from exc
    # Return both types so the rest of the file stays decoupled from the lazy import.
    return DocumentInfo, MossClient


def normalize_text(text: str) -> str:
    # Collapse long vertical gaps so chunking is based on content, not accidental spacing.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces and tabs while preserving paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    # Remove leading/trailing whitespace before splitting into chunks.
    return text.strip()


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    # Normalize first so the chunker sees consistent whitespace across all files.
    text = normalize_text(text)
    if len(text) <= chunk_chars:
        # Small documents stay as one chunk; splitting them would only remove context.
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        # Start with a hard maximum window, then move the split backward if a
        # better markdown boundary exists.
        end = min(start + chunk_chars, len(text))
        # Prefer splitting at second-level headings, which usually begin topics.
        boundary = text.rfind("\n## ", start, end)
        if boundary <= start + 400:
            # If no useful heading exists, split at a paragraph boundary.
            boundary = text.rfind("\n\n", start, end)
        if boundary <= start + 400:
            # If neither boundary is useful, fall back to the maximum chunk size.
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        if boundary >= len(text):
            break
        # Step back so adjacent chunks share context around the split.
        start = max(0, boundary - CHUNK_OVERLAP)

    return chunks


def index_info_matches_name(index_info, index_name: str) -> bool:
    # Moss clients may return strings, dicts, or typed objects depending on version.
    # This helper keeps the script tolerant of those response shapes.
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


async def moss_index_exists(client, index_name: str) -> bool:
    # Check existing indexes first, which makes this script safe to rerun.
    indexes = await client.list_indexes()
    return any(index_info_matches_name(index_info, index_name) for index_info in indexes)


def document_id(document) -> str | None:
    if isinstance(document, dict):
        return document.get("id")
    return getattr(document, "id", None)


async def sync_existing_moss_index(client, index_name: str, documents) -> None:
    # Fetch current remote chunks so this run can remove content that no longer
    # exists locally.
    remote_documents = await client.get_docs(name=index_name)
    local_ids = {document_id(document) for document in documents}
    remote_ids = {document_id(document) for document in remote_documents}
    local_ids.discard(None)
    remote_ids.discard(None)

    # Anything present remotely but absent locally is stale after a file edit or deletion.
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
    # add_docs handles both new chunks and updated chunks with the same IDs.
    await client.add_docs(name=index_name, docs=documents)

def load_documents():
    # DocumentInfo is the object Moss expects for each searchable chunk.
    DocumentInfo, _ = import_moss_client_types()
    markdown_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not markdown_files:
        raise RuntimeError(f"No markdown files found in {KNOWLEDGE_DIR}")

    documents = []
    for path in markdown_files:
        # Read markdown as UTF-8 so the public repo behaves consistently across OSes.
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text), start=1):
            # Include chunk text in the digest so edits produce new IDs and stale
            # remote chunks can be detected.
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


async def create_index() -> None:
    # Load .env first and .env.local second so local overrides win in development.
    load_dotenv(PROJECT_DIR / ".env", override=True)
    load_dotenv(PROJECT_DIR / ".env.local", override=True)
    require_env("MOSS_PROJECT_ID", "MOSS_PROJECT_KEY")

    _, MossClient = import_moss_client_types()
    # These environment variables connect the offline indexer to the later runtime
    # retrieval stage. Defaults keep the first run simple.
    index_name = os.getenv("MOSS_INDEX_NAME", DEFAULT_INDEX_NAME)
    model_id = os.getenv("MOSS_MODEL_ID", "moss-minilm")
    client = MossClient(
        project_id=os.environ["MOSS_PROJECT_ID"],
        project_key=os.environ["MOSS_PROJECT_KEY"],
    )

    documents = load_documents()
    if await moss_index_exists(client, index_name):
        logger.info(
            "Moss index '{}' already exists; diff-syncing {} local chunks",
            index_name,
            len(documents),
        )
        # Existing indexes are synchronized instead of recreated so this command is
        # repeatable while iterating on knowledge-base markdown files.
        await sync_existing_moss_index(client, index_name, documents)
        logger.success("Moss index '{}' synced successfully", index_name)
    else:
        logger.info("Uploading {} chunks to Moss index '{}'", len(documents), index_name)
        # First-time setup creates the remote Moss index and uploads all chunks.
        await client.create_index(name=index_name, docs=documents, model_id=model_id)
        logger.success("Moss index '{}' created successfully", index_name)
    logger.info("Use MOSS_INDEX_NAME={} when running Stage 07 and later", index_name)


if __name__ == "__main__":
    asyncio.run(create_index())



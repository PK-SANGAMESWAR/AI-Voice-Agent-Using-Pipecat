"""Optional Moss retrieval wiring for the final voice agent.

The community `pipecat-moss` package currently targets older Pipecat frame names.
This module keeps the same idea but implements the tiny processor locally against
Pipecat's `LLMContextFrame`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

MOSS_REQUIRED_ENV = ("MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "MOSS_INDEX_NAME")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def moss_configured() -> bool:
    """Return whether all Moss runtime variables are present."""
    return all(os.getenv(name) for name in MOSS_REQUIRED_ENV)


class MossContextProcessor(FrameProcessor):
    """Retrieve Moss passages and append them to each emitted LLM context."""

    def __init__(
        self,
        *,
        client: Any,
        index_name: str,
        top_k: int,
        alpha: float,
        system_prompt: str,
    ) -> None:
        super().__init__(name=f"MossRetrieval-{index_name}")
        self._client = client
        self._index_name = index_name
        self._top_k = top_k
        self._alpha = alpha
        self._system_prompt = system_prompt
        self._last_query: str | None = None
        self._injected_message: dict[str, Any] | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        try:
            messages = frame.context.get_messages()
            latest_user_text = self._latest_user_text(messages)
            if not latest_user_text:
                await self.push_frame(frame, direction)
                return

            if latest_user_text == self._last_query:
                logger.debug("{}: skipping duplicate Moss query", self)
                await self.push_frame(frame, direction)
                return

            self._last_query = latest_user_text
            search_result = await self._retrieve(latest_user_text)
            documents = getattr(search_result, "docs", []) or []
            logger.debug("{}: Moss returned {} documents", self, len(documents))
            # Keep only the passages for the current turn; otherwise every turn
            # appends another system message and the context grows without bound.
            self._remove_injected_message(frame.context)
            if documents:
                self._injected_message = {
                    "role": "system",
                    "content": self._format_documents(documents),
                }
                frame.context.add_message(self._injected_message)
            await self.push_frame(frame, direction)
        except Exception as exc:
            logger.exception("{}: Moss retrieval failed: {}", self, exc)
            await self.push_error(f"{self} retrieval error: {exc}", exception=exc)
            await self.push_frame(frame, direction)

    def _remove_injected_message(self, context) -> None:
        previous = self._injected_message
        self._injected_message = None
        if previous is None:
            return
        messages = context.get_messages()
        remaining = [message for message in messages if message is not previous]
        if len(remaining) != len(messages):
            context.set_messages(remaining)

    async def _retrieve(self, query: str):
        from inferedge_moss import QueryOptions

        result = await self._client.query(
            self._index_name,
            query,
            options=QueryOptions(top_k=self._top_k, alpha=self._alpha),
        )
        if self.metrics_enabled:
            time_taken_ms = getattr(result, "time_taken_ms", None)
            if time_taken_ms is not None:
                await self.push_frame(
                    MetricsFrame(
                        data=[
                            ProcessingMetricsData(
                                processor=self.name,
                                value=time_taken_ms / 1000.0,
                            )
                        ]
                    )
                )
        return result

    @staticmethod
    def _latest_user_text(messages: Sequence[dict[str, Any]]) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ).strip()
        return None

    def _format_documents(self, documents: Sequence[Any]) -> str:
        lines = [self._system_prompt.rstrip(), ""]
        for index, document in enumerate(documents, start=1):
            metadata = getattr(document, "metadata", None) or {}
            extras = []
            if source := metadata.get("source"):
                extras.append(f"source={source}")
            if score := getattr(document, "score", None):
                extras.append(f"score={score}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"{index}. {getattr(document, 'text', '') or ''}{suffix}")
        return "\n".join(lines).strip()


async def make_moss_retrieval_processor() -> FrameProcessor | None:
    """Load the configured Moss index and return a Pipecat processor."""
    missing = [name for name in MOSS_REQUIRED_ENV if not os.getenv(name)]
    if missing:
        logger.info("Moss retrieval disabled; missing env vars: {}", ", ".join(missing))
        return None

    try:
        from inferedge_moss import MossClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Moss env vars are configured, but inferedge-moss is not installed. "
            "Run `uv sync --active` or `pip install pipecat-moss`."
        ) from exc

    index_name = os.environ["MOSS_INDEX_NAME"]
    client = MossClient(
        project_id=os.environ["MOSS_PROJECT_ID"],
        project_key=os.environ["MOSS_PROJECT_KEY"],
    )
    await client.load_index(index_name)
    logger.info("Loaded Moss index: {}", index_name)
    return MossContextProcessor(
        client=client,
        index_name=index_name,
        top_k=_env_int("MOSS_TOP_K", 5),
        alpha=_env_float("MOSS_ALPHA", 0.8),
        system_prompt="Relevant passages from the CampusX Moss knowledge base:\n\n",
    )
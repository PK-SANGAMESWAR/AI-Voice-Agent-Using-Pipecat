"""Shared Pipecat utilities used by final and staged agents."""

from __future__ import annotations

import os
from collections.abc import Sequence

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

TRANSPORT_PARAMS = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def require_env(*names: str) -> None:
    """Fail with a clear setup message instead of an obscure SDK error."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing environment variable(s): {joined}. "
            "Copy values from .env.example into a local .env.local file."
        )


def make_user_turn_params(*, interruptions: bool = True) -> LLMUserAggregatorParams:
    """Small local VAD + timers; no local STT, LLM, TTS, or Smart Turn model."""
    strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=interruptions),
            TranscriptionUserTurnStartStrategy(enable_interruptions=interruptions),
        ],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)],
    )
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_turn_strategies=strategies,
    )


class MetricsPrinter(FrameProcessor):
    """Make service latency and usage MetricsFrames readable in the terminal."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for metric in frame.data:
                fields = ", ".join(
                    f"{name}={value!r}"
                    for name, value in vars(metric).items()
                    if not name.startswith("_")
                )
                logger.info("METRIC {}: {}", type(metric).__name__, fields)
        await self.push_frame(frame, direction)


async def make_transport(runner_args: RunnerArguments) -> BaseTransport:
    """Create the requested WebRTC transport from Pipecat runner args."""
    return await create_transport(runner_args, TRANSPORT_PARAMS)


async def run_pipeline(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    processors: Sequence[FrameProcessor],
    *,
    context: LLMContext | None = None,
    metrics: bool = False,
) -> None:
    """Run a pipeline and wire common connect/disconnect lifecycle events."""
    pipeline = Pipeline(list(processors))
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=metrics,
            enable_usage_metrics=metrics,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Browser connected: {}", client)

    # If the browser tab closes, stop the whole pipeline cleanly.
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Browser disconnected: {}", client)
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint, force_gc=True)
    await runner.add_workers(worker)
    await runner.run()
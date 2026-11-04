"""Stage 04: add an LLM and print its text; it has no business identity yet.

The text LLM answers a complete user turn, but only as a generic assistant.
The next stage gives it a real role through the system prompt.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
# Stage 04's new service is the text LLM. It consumes the completed user turn
# from LLMContext and emits assistant text Frames.
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesLLMService,
    OpenAIResponsesReasoningConfig,
)
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner



def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing environment variable(s): {joined}. "
            "Copy the value into your local environment before running."
        )


async def make_transport(runner_args: RunnerArguments) -> BaseTransport:
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    return await create_transport(runner_args, transport_params)


async def run_pipeline(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    processors: Sequence[FrameProcessor],
    *,
    context: LLMContext | None = None,
    metrics: bool = False,
) -> None:
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Browser disconnected: {}", client)
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint, force_gc=True)
    await runner.add_workers(worker)
    await runner.run()


class DebugFramePrinter(FrameProcessor):

    def __init__(self, label: str, *, audio_every: int = 100):
        super().__init__()
        self._label = label
        self._audio_every = audio_every
        self._input_audio_count = 0
        self._output_audio_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._input_audio_count += 1
            if self._input_audio_count == 1 or self._input_audio_count % self._audio_every == 0:
                logger.info(
                    "{} -> {} #{} ({} bytes, {} Hz)",
                    self._label,
                    type(frame).__name__,
                    self._input_audio_count,
                    len(frame.audio),
                    frame.sample_rate,
                )
        elif isinstance(frame, OutputAudioRawFrame):
            self._output_audio_count += 1
            if self._output_audio_count == 1 or self._output_audio_count % self._audio_every == 0:
                logger.info(
                    "{} -> {} #{} ({} bytes, {} Hz)",
                    self._label,
                    type(frame).__name__,
                    self._output_audio_count,
                    len(frame.audio),
                    frame.sample_rate,
                )
        else:
            text = getattr(frame, "text", None)
            suffix = f": {text}" if text else ""
            logger.info("{} -> {}{}", self._label, type(frame).__name__, suffix)

        await self.push_frame(frame, direction)


def make_user_turn_params(*, interruptions: bool = False) -> LLMUserAggregatorParams:
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


# ---------------------------------------------------------------------------
# Stage 04 bot: intelligence arrives, but it has no identity yet.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    require_env("OPENAI_API_KEY")

    stt = OpenAIRealtimeSTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        ),
    )

    # Stage 04 introduces generation. The LLM receives the completed user turn
    # from context and produces assistant text, but no CampusX identity exists yet.
    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            # The model is configurable through the environment for demos and tests.
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
            # Minimal reasoning keeps this early text-only stage responsive.
            reasoning=OpenAIResponsesReasoningConfig(effort="minimal"),
        )
    )

    context = LLMContext()

    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=make_user_turn_params(),
    )

    await run_pipeline(
        transport,
        runner_args,
        [
            transport.input(),
            stt,
            user_aggregator,
            # The new LLM processor is placed after the user aggregator so it sees
            # complete turns rather than partial transcript chunks.
            llm,
            DebugFramePrinter("OPENAI TEXT"),
        ],
        # Bind the LLM service to the conversation state populated by the user aggregator.
        context=context,
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
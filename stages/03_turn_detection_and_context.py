"""Stage 03: add Silero VAD, explicit turn boundaries, and the user context.

Local VAD plus turn strategies let the user aggregator commit one complete user
turn into shared conversation context instead of feeding raw transcript chunks.
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
    """Configure the turn-boundary behavior introduced in Stage 03."""
    # Stage 02 printed transcript fragments. Stage 03 groups those fragments into
    # one conversational turn so later LLM stages receive complete messages.
    strategies = UserTurnStrategies(
        start=[
            # VAD notices real speech energy from the microphone before a full
            # transcript is available.
            VADUserTurnStartStrategy(enable_interruptions=interruptions),
            # Transcription start events provide a second signal from the STT layer,
            # making turn starts more robust than VAD alone.
            TranscriptionUserTurnStartStrategy(enable_interruptions=interruptions),
        ],
        # After 0.6 seconds of silence, commit the user turn to LLMContext.
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)],
    )
    return LLMUserAggregatorParams(
        # Silero runs locally and supplies the speech/silence signal used above.
        vad_analyzer=SileroVADAnalyzer(),
        # Pass the explicit start/stop rules into the user aggregator.
        user_turn_strategies=strategies,
    )


# ---------------------------------------------------------------------------
# Stage 03 bot: one complete user turn, still nothing answers.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    require_env("OPENAI_API_KEY")

    stt = OpenAIRealtimeSTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        ),
    )

    # Stage 03 introduces shared conversation state. The user aggregator writes
    # completed user turns into this context; no assistant messages exist yet.
    context = LLMContext()

    # LLMContextAggregatorPair returns both sides of conversation memory. This
    # stage uses only the user side and ignores the assistant side until Stage 09.
    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        # The new turn params tell the aggregator when to stop collecting STT text
        # and when to commit one complete user message.
        user_params=make_user_turn_params(),
    )

    # This event is the key checkpoint for Stage 03: it fires only after the
    # aggregator has decided the user turn is finished.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        logger.info("USER TURN COMPLETE: {}", message.content)

    await run_pipeline(
        transport,
        runner_args,
        [
            transport.input(),
            stt,
            # The new user aggregator sits after STT because it needs text
            # fragments before it can form one complete user message.
            user_aggregator,
            DebugFramePrinter("AFTER USER AGGREGATOR"),
        ],
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
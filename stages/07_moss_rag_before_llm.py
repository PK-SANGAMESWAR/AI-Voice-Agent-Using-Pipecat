"""Stage 07: retrieve Moss passages before the LLM answers.

Moss is a processor in the LLM knowledge path, not a function tool. The pipeline
stays text-only so the retrieval step is isolated before generation.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

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

# Stage 07 is the first voice-pipeline stage that reuses runtime package code
# from src/. This keeps the standalone stage runnable without installation.
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# New runtime component: a Pipecat processor that retrieves Moss passages and
# injects them into LLMContext before generation.
from campusx_voice_agent.moss import make_moss_retrieval_processor


# Stage 07 extends the existing prompt with the Moss-grounding rule near the end.
GUIDANCE_SYSTEM_PROMPT = """
You are the CampusX free-guidance-session booking assistant on a live voice call.
Your job is to help a caller book a free 20-minute guidance session.
Speak in short, warm, natural sentences without markdown.

Collect these details one at a time: full name, email address, learning or career goal,
and preferred day or time of day. Use the availability tool before offering slots.
Offer no more than three available slots at once.

Before booking, repeat the name, email, learning goal, and exact slot, then ask for an
explicit yes or no. Call the booking tool only after the caller says yes. Never invent
a slot or booking ID. If an email looks unclear, ask them to spell it.

Use retrieved Moss knowledge-base context when answering questions about CampusX
courses, schedules, curriculum, or policies. If Moss context is absent or not
relevant, say you do not know instead of inventing facts.
""".strip()


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
# Stage 07 bot: Moss enriches each user turn before text generation.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # Stage 07 adds Moss runtime credentials because the live pipeline now calls
    # the remote Moss index created in Stage 06.
    require_env("OPENAI_API_KEY", "MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "MOSS_INDEX_NAME")

    stt = OpenAIRealtimeSTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        ),
    )

    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
            reasoning=OpenAIResponsesReasoningConfig(effort="minimal"),
            system_instruction=GUIDANCE_SYSTEM_PROMPT,
        ),
    )

    context = LLMContext()

    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=make_user_turn_params(),
    )

    # Build the retrieval processor after env validation. It reads Moss settings
    # and returns None when retrieval is not configured.
    moss_retrieval = await make_moss_retrieval_processor()
    if moss_retrieval is None:
        # This stage is specifically about Moss, so fail loudly instead
        # of silently falling back to an ungrounded LLM.
        raise RuntimeError("Moss retrieval is required for Stage 07.")

    await run_pipeline(
        transport,
        runner_args,
        [
            transport.input(),
            stt,
            user_aggregator,
            # New ordering: retrieve after the completed user turn is known, but
            # before the LLM starts answering.
            moss_retrieval,
            llm,
            DebugFramePrinter("MOSS-GROUNDED TEXT"),
        ],
        context=context,
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

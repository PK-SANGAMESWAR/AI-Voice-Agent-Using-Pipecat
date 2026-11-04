"""Stage 02: add cloud STT and print transcripts; there is still no agent response.

OpenAI streaming speech-to-text converts incoming audio Frames into text
transcription Frames. The pipeline can now understand words, but nothing answers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
# Stage 02's new processor: OpenAI streaming STT converts raw audio Frames into
# transcription Frames while the user is speaking.
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext



def require_env(*names: str) -> None:
    # Stage 02 is the first cloud-backed stage, so fail before constructing
    # OpenAI services if required environment values are missing.
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        # Join all missing names so the user can fix every environment value
        # in one pass instead of discovering them one failure at a time.
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


# ---------------------------------------------------------------------------
# Stage 02 bot: audio becomes text, nothing responds yet.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # The OpenAI key must exist before this stage can create the STT service.
    require_env("OPENAI_API_KEY")

    # This is the new Stage 02 processor. It receives InputAudioRawFrame objects
    # from transport.input() and emits transcription Frames containing text.
    stt = OpenAIRealtimeSTTService(
        # Use the same key for OpenAI's realtime transcription service.
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(
            # Keep the model configurable so users can switch STT models without
            # editing code during deployment or experimentation.
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        ),
    )

    await run_pipeline(
        transport,
        runner_args,
        [
            transport.input(),
            # Insert STT immediately after raw microphone input, so the printer
            # now shows transcription Frames instead of only audio Frames.
            stt,
            DebugFramePrinter("AFTER OPENAI STT"),
        ],
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
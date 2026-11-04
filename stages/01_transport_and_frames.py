"""Stage 01: audio reaches Python as Pipecat Frames, but it has no meaning yet.

Adds the local WebRTC transport and a debug frame printer so we can
see raw audio frames arriving from the browser microphone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

# LLMContext is not used until later stages. TYPE_CHECKING keeps the type hint available
# to editors without importing the aggregator module at runtime here.
if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext

# ---------------------------------------------------------------------------
# Stage 01 introduces these helpers. Later files reuse them without re-explaining
# them, so the comments live here where the transport and pipeline first appear.
# ---------------------------------------------------------------------------


async def make_transport(runner_args: RunnerArguments) -> BaseTransport:
    """Build the local WebRTC transport that connects the browser to Pipecat."""
    # Pipecat's development runner asks for a transport by name. For local browser
    # demos that transport is WebRTC, so the key here is "webrtc".
    transport_params = {
        # The lambda delays TransportParams construction until create_transport knows
        # which runner transport it is building.
        "webrtc": lambda: TransportParams(
            # Enable microphone audio from the browser into the Python pipeline.
            audio_in_enabled=True,
            # Enable speaker audio back to the browser for later TTS stages.
            audio_out_enabled=True,
        ),
    }
    # create_transport returns a BaseTransport with input() and output() processors.
    return await create_transport(runner_args, transport_params)


async def run_pipeline(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    processors: Sequence[FrameProcessor],
    *,
    context: LLMContext | None = None,
    metrics: bool = False,
) -> None:
    """Run one stage pipeline and wire connect/disconnect lifecycle events."""
    # A Pipeline is an ordered list: every Frame emitted by one processor is
    # pushed to the next processor in this exact sequence.
    pipeline = Pipeline(list(processors))
    # PipelineWorker owns the async lifecycle of that Pipeline. Later stages reuse
    # this helper so their examples can focus on the new processor they add.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Keep metrics configurable; this first stage does not need them, but
            # the helper is ready for later latency and usage demonstrations.
            enable_metrics=metrics,
            enable_usage_metrics=metrics,
        ),
        # Reuse the runner timeout so a stuck browser session does not leave the
        # demo worker alive forever.
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # The transport emits this when the browser tab joins the WebRTC session.
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Browser connected: {}", client)

    # The disconnect handler is the cleanup path for a live browser demo: close
    # the browser tab and the pipeline stops instead of hanging in the terminal.
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Browser disconnected: {}", client)
        await worker.cancel()

    # WorkerRunner connects Pipecat's runner arguments to the worker lifecycle and
    # handles Ctrl+C consistently while the demo is running.
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint, force_gc=True)
    await runner.add_workers(worker)
    await runner.run()


class DebugFramePrinter(FrameProcessor):
    """Print important frames while throttling high-frequency audio packets."""

    def __init__(self, label: str, *, audio_every: int = 100):
        super().__init__()
        # A label makes it clear where this printer sits in the processor chain.
        self._label = label
        # Audio frames are frequent, so sample logs after the first frame.
        self._audio_every = audio_every
        # Keep input and output counts separate for later TTS stages.
        self._input_audio_count = 0
        self._output_audio_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Always call the base processor first so Pipecat can maintain normal
        # frame-processing bookkeeping.
        await super().process_frame(frame, direction)

        # InputAudioRawFrame is the proof that microphone bytes reached Python.
        # Log the first packet, then every N-th packet, to avoid flooding the terminal.
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
        # OutputAudioRawFrame is unused in Stage 01, but supporting it now makes
        # this printer reusable when TTS appears later.
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
            # For non-audio Frames, include text when the frame exposes it. Later
            # STT and LLM stages benefit from this same printer.
            text = getattr(frame, "text", None)
            suffix = f": {text}" if text else ""
            logger.info("{} -> {}{}", self._label, type(frame).__name__, suffix)

        # A processor must push the frame onward. Without this line the printer
        # would swallow the frame and stop the pipeline.
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Stage 01 bot: raw audio in, no meaning yet.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # Stage 01 has the smallest useful pipeline: browser microphone input first,
    # then a printer that proves those audio Frames reached Python.
    await run_pipeline(
        transport,
        runner_args,
        [
            # transport.input() is the source processor for browser microphone Frames.
            transport.input(),
            # This stage stops at observation; it does not transcribe or answer yet.
            DebugFramePrinter("RAW INPUT"),
        ],
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
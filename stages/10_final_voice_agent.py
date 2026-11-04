"""Stage 10: final Moss-grounded CampusX booking voice agent.

This combines booking tools with real browser TTS, conversation memory, Moss retrieval,
and barge-in interruption handling.
"""

from __future__ import annotations

import os
import sys
# Stage 10 adds these standard-library modules for booking tools: regex email
# validation, SQLite persistence, and short public booking IDs.
import re
import sqlite3
import uuid
from collections.abc import Sequence
# The rolling availability calendar is generated from dates and times at runtime.
from datetime import date, datetime, time, timedelta
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    # Stage 10 queues a short spoken filler frame while a tool call is running.
    TTSSpeakFrame,
)
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
from pipecat.services.openai.tts import OpenAITTSService
# FunctionCallParams gives each tool access to result_callback, the bridge back
# from deterministic Python code to the LLM.
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from campusx_voice_agent.moss import make_moss_retrieval_processor


# Stage 10 introduces local persistence for real bookings, so the stage needs a
# stable data directory next to the knowledge-base files.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Stage 10 keeps the existing identity prompt and pairs it with real function
# tools below. The prompt still tells the LLM when it may call those tools.
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


def make_user_turn_params(*, interruptions: bool = True) -> LLMUserAggregatorParams:
    # Stage 10 changes the default to interruptions=True. The same turn logic
    # now also lets the user interrupt an assistant while it is speaking.
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


# --- Booking tools: real availability reads and SQLite booking writes. -------

# Stage 10 adds the first action tools. This regex keeps obviously invalid email
# strings out of the bookings database before a confirmation is saved.
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
# The demo availability model is intentionally small and deterministic: three
# possible guidance slots per eligible day.
_SLOT_TIMES = (time(11, 0), time(16, 0), time(19, 0))


def _database_path() -> Path:
    # Allow deployments or local demos to move the database without editing code.
    configured = os.getenv("GUIDANCE_DB_PATH")
    # Fall back to data/guidance_bookings.db so the public repo works out of the box.
    return Path(configured) if configured else DATA_DIR / "guidance_bookings.db"


def _connect() -> sqlite3.Connection:
    path = _database_path()
    # Create the data folder lazily so a fresh clone can book a session immediately.
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    # Row objects let later queries access columns by name instead of index.
    connection.row_factory = sqlite3.Row
    # Create the bookings table on first use. The slot is UNIQUE so two callers
    # cannot reserve the same guidance time.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id TEXT PRIMARY KEY,
            caller_name TEXT NOT NULL,
            email TEXT NOT NULL,
            learning_goal TEXT NOT NULL,
            slot TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    return connection


def _future_slots(days: int = 7) -> list[tuple[datetime, str]]:
    # Generate a rolling list of possible slots instead of hard-coding dates, so
    # the demo remains usable after the repository is published.
    slots: list[tuple[datetime, str]] = []
    today = date.today()
    for day_offset in range(1, days + 1):
        slot_date = today + timedelta(days=day_offset)
        # Skip Sundays for this demo availability policy.
        if slot_date.weekday() == 6:
            continue
        for slot_time in _SLOT_TIMES:
            value = datetime.combine(slot_date, slot_time)
            # The label is exactly what the LLM must repeat back before booking.
            label = value.strftime("%A, %d %B %Y at %I:%M %p IST")
            slots.append((value, label))
    return slots


def available_guidance_slots(time_preference: str = "any", limit: int = 6) -> list[str]:
    # Normalize the caller preference so simple words like "evening" are enough.
    preference = time_preference.strip().lower()
    with _connect() as connection:
        # Read existing bookings first; those labels are removed from the offer list.
        booked = {row["slot"] for row in connection.execute("SELECT slot FROM bookings")}

    candidates = _future_slots()
    # Filter by broad time-of-day language; deterministic business logic belongs
    # in tools, not in the model response.
    if "morning" in preference:
        candidates = [item for item in candidates if item[0].hour < 12]
    elif "afternoon" in preference:
        candidates = [item for item in candidates if 12 <= item[0].hour < 18]
    elif "evening" in preference:
        candidates = [item for item in candidates if item[0].hour >= 18]

    # Return labels because the LLM should offer and later book exact labels.
    return [label for _, label in candidates if label not in booked][:limit]


async def get_available_guidance_slots(
    params: FunctionCallParams,
    time_preference: str,
):
    """Find currently available free-guidance slots.

    Args:
        time_preference: Caller preference such as morning, afternoon, evening, or any.
    """
    # Call deterministic Python logic, then return structured data to the LLM.
    slots = available_guidance_slots(time_preference)
    await params.result_callback({"available_slots": slots, "timezone": "Asia/Kolkata"})


async def book_free_guidance_session(
    params: FunctionCallParams,
    caller_name: str,
    email: str,
    learning_goal: str,
    preferred_slot: str,
):
    """Book a free guidance session after the caller explicitly confirms all details.

    Args:
        caller_name: Caller's confirmed full name.
        email: Caller's confirmed email address.
        learning_goal: What the caller wants guidance about.
        preferred_slot: Exact slot returned by get_available_guidance_slots.
    """
    # Normalize caller-provided fields before validation or storage.
    name = caller_name.strip()
    normalized_email = email.strip().lower().replace(" ", "")
    goal = learning_goal.strip()
    # Only exact labels generated by availability can be booked.
    valid_slots = {label for _, label in _future_slots()}

    if len(name) < 2:
        result = {"booked": False, "error": "A full name is required."}
    elif not _EMAIL_PATTERN.match(normalized_email):
        result = {"booked": False, "error": "The email address is not valid."}
    elif len(goal) < 3:
        result = {"booked": False, "error": "A learning goal is required."}
    elif preferred_slot not in valid_slots:
        # This prevents the model from inventing a time that was never offered.
        result = {"booked": False, "error": "Choose an exact slot returned by availability."}
    else:
        # Generate a short public-facing booking ID after all validation passes.
        booking_id = f"CX-{uuid.uuid4().hex[:8].upper()}"
        try:
            with _connect() as connection:
                # Use parameterized SQL so caller values are never formatted directly
                # into the query string.
                connection.execute(
                    """
                    INSERT INTO bookings (
                        booking_id, caller_name, email, learning_goal, slot, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        booking_id,
                        name,
                        normalized_email,
                        goal,
                        preferred_slot,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ),
                )
            result = {
                "booked": True,
                "booking_id": booking_id,
                "caller_name": name,
                "email": normalized_email,
                "learning_goal": goal,
                "slot": preferred_slot,
            }
        except sqlite3.IntegrityError:
            # The UNIQUE slot constraint lands here if another booking takes the
            # same slot between availability lookup and insert.
            result = {
                "booked": False,
                "error": "That slot was just booked. Please check availability again.",
            }

    # Send the final structured result back to the LLM so it can speak success or
    # the validation error naturally to the caller.
    await params.result_callback(result)



# Register exactly the two Stage 10 action tools the LLM is allowed to call.
GUIDANCE_TOOLS = [get_available_guidance_slots, book_free_guidance_session]


# ---------------------------------------------------------------------------
# Stage 10 bot: final booking agent with barge-in support.
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
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

    tts = OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAITTSService.Settings(
            model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
            voice=os.getenv("OPENAI_TTS_VOICE", "nova"),
        ),
    )

    # Stage 10 improves live-call behavior during tool execution. When the LLM
    # starts a function call, queue a short spoken acknowledgement immediately.
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(
            # append_to_context=False keeps this filler sentence out of long-term
            # memory; it is playback UX, not assistant reasoning.
            TTSSpeakFrame("One moment while I check that.", append_to_context=False)
        )

    # Passing tools into LLMContext exposes the Stage 10 booking functions to the
    # model while preserving the same pipeline shape from Stage 09.
    context = LLMContext(tools=GUIDANCE_TOOLS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # interruptions=True broadcasts an InterruptionFrame as soon as the user
        # starts speaking, cancelling in-flight LLM generation and TTS playback.
        user_params=make_user_turn_params(interruptions=True),
    )

    moss_retrieval = await make_moss_retrieval_processor()
    if moss_retrieval is None:
        raise RuntimeError("Moss retrieval is required for this stage.")

    await run_pipeline(
        transport,
        runner_args,
        [
            transport.input(),
            # This Stage 10 printer is before STT so interruption-related Frames
            # can be seen as soon as the user speaks over the assistant.
            DebugFramePrinter("TURN / INTERRUPTION"),
            stt,
            user_aggregator,
            moss_retrieval,
            llm,
            tts,
            DebugFramePrinter("GENERATED TTS AUDIO", audio_every=50),
            transport.output(),
            assistant_aggregator,
        ],
        context=context,
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()


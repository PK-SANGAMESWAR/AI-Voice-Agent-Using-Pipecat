"""Final local CampusX guidance-booking voice agent used for the live demo."""

from __future__ import annotations

import os

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesLLMService,
    OpenAIResponsesReasoningConfig,
)
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport

from campusx_voice_agent.moss import make_moss_retrieval_processor
from campusx_voice_agent.pipeline import (
    MetricsPrinter,
    make_transport,
    make_user_turn_params,
    require_env,
    run_pipeline,
)
from campusx_voice_agent.tools import GUIDANCE_SYSTEM_PROMPT, GUIDANCE_TOOLS


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    require_env("OPENAI_API_KEY")

    # llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-5")


    stt = OpenAIRealtimeSTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeSTTService.Settings(
            # model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
            model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe"),
        ),
    )
    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            model=llm_model,
            system_instruction=GUIDANCE_SYSTEM_PROMPT,
            reasoning=OpenAIResponsesReasoningConfig(effort="minimal"),
        ),
    )
    tts = OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAITTSService.Settings(
            model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
            voice=os.getenv("OPENAI_TTS_VOICE", "nova"),
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(
            TTSSpeakFrame("One moment while I check that.", append_to_context=False)
        )

    context = LLMContext(tools=GUIDANCE_TOOLS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=make_user_turn_params(interruptions=True),
    )
    moss_retrieval = await make_moss_retrieval_processor()

    processors = [
        transport.input(),
        stt,
        user_aggregator,
    ]
    if moss_retrieval is not None:
        processors.append(moss_retrieval)
    processors.extend(
        [
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
            MetricsPrinter(),
        ]
    )

    await run_pipeline(
        transport,
        runner_args,
        processors,
        context=context,
        metrics=True,
    )


async def bot(runner_args: RunnerArguments):
    """WebRTC bot entry point used by Pipecat's development runner."""
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)

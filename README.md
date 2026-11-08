# CampusX Voice Agent with Pipecat

A real-time browser voice agent that answers questions about CampusX learning resources and can book free guidance sessions. It uses Pipecat for the WebRTC audio pipeline, OpenAI for speech recognition, responses, and speech synthesis, a local Markdown knowledge base, and SQLite for bookings. Moss semantic retrieval can be enabled as an additional knowledge source.

## What it can do

- Talk with a caller through a browser microphone and speakers.
- Answer CampusX questions using verified local knowledge-base files, with optional Moss retrieval.
- Find available free 20-minute guidance-session slots and save confirmed bookings in SQLite.
- Keep the conversation interruptible and print pipeline metrics in the terminal.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI API key
- For optional Moss retrieval and indexing: Moss project credentials

## Setup

From the repository root, install the project dependencies:

```powershell
uv sync
```

Create a local environment file from the template and add your credentials:

```powershell
Copy-Item .env.example .env.local
```

At minimum, set `OPENAI_API_KEY` in `.env.local`. The app loads `.env` and then `.env.local`, with `.env.local` taking precedence. Keep credentials out of source control.

The main agent defaults to `gpt-4o-transcribe` for speech recognition, `gpt-5` for responses, and `tts-1` with the `nova` voice for speech synthesis. You can override these in `.env.local`:

```dotenv
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_LLM_MODEL=gpt-5
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=nova
```

## Run the voice agent

Start the development runner:

```powershell
uv run python -m campusx_voice_agent.agent
```

Pipecat's runner finds the `bot()` function on the module that is executed, so run the agent module itself. Running `python -m pipecat.runner.run` directly does not work: it accepts no bot-file argument, and every session fails with "Could not find 'bot' function". Runner options such as `--port 7861` can be appended.

Open the local URL printed by the runner (default `http://localhost:7860`) in a browser, allow microphone access, and start a conversation. The agent requires `OPENAI_API_KEY`. If Moss credentials are missing, it continues without semantic retrieval and uses the local knowledge-base search tool when it needs CampusX facts.

## Knowledge sources

The local knowledge base lives in [`data/knowledge_base`](data/knowledge_base). It currently contains CampusX course curriculum and schedule material. The agent searches Markdown and text files in this directory. To use a different directory, set `CAMPUSX_KNOWLEDGE_BASE_DIR`.

Moss retrieval is optional at runtime. To enable it, set these values in `.env.local`:

```dotenv
MOSS_PROJECT_ID=your-project-id
MOSS_PROJECT_KEY=your-project-key
MOSS_INDEX_NAME=campusx-voice-agent-knowledge
```

Create or sync the Moss index from the local Markdown files with:

```powershell
uv run python -m campusx_voice_agent.moss_index
```

The index name defaults to `campusx-voice-agent-knowledge`; `MOSS_INDEX_NAME` can override it. Indexing also accepts `MOSS_MODEL_ID` to choose the Moss embedding model (default `moss-minilm`).

## Guidance-session bookings

The agent asks for the caller’s name, email, learning goal, and preferred time, checks availability, and only books after explicit confirmation. Bookings are stored in `data/guidance_bookings.db` by default. Set `GUIDANCE_DB_PATH` to use another SQLite file.

Available slots are generated for the next seven days, excluding Sundays, at 11:00 AM, 4:00 PM, and 7:00 PM India Standard Time. Existing reservations are excluded. Booking data is local to this app; it is not sent to a scheduling service.

To view saved bookings:

```powershell
uv run python -m campusx_voice_agent.bookings
```

## Learning stages

The [`stages`](stages) directory contains numbered, progressively built examples, from an empty project through transport and audio frames, STT, turn handling, LLM responses, TTS, Moss retrieval, conversation memory, and the final booking agent. Run a stage directly, for example `uv run python stages/03_turn_detection_and_context.py`. Stages 07–10 require Moss credentials and an index built with stage 06 or `campusx_voice_agent.moss_index`. The integrated package implementation is in [`src/campusx_voice_agent`](src/campusx_voice_agent).

## Project layout

```text
data/knowledge_base/       Local CampusX reference documents
src/campusx_voice_agent/   Integrated agent, pipeline, search, Moss, and booking code
stages/                    Numbered Pipecat learning examples
```

## Troubleshooting

- **"Failed to start session" in the browser:** Start the agent with `uv run python -m campusx_voice_agent.agent`, not `python -m pipecat.runner.run`. If it still fails, check the terminal for an OpenAI `401`/`expired_secret_key` error, which means the API key must be replaced.
- **Missing `OPENAI_API_KEY`:** Add it to `.env.local` and restart the runner.
- **Irrelevant answers when Moss is enabled:** `MOSS_INDEX_NAME` must point at an index built from `data/knowledge_base` (default `campusx-voice-agent-knowledge`). Retrieved passages are injected into every turn.
- **Moss retrieval unavailable:** Check the three Moss environment variables and confirm the configured index exists. The agent can still run without Moss.
- **Browser has no audio:** Allow microphone access and use a browser that supports WebRTC.
- **No verified answer found:** Add or update the relevant Markdown or text document under `data/knowledge_base`.


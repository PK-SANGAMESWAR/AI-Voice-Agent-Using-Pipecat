"""Real local actions for the CampusX free-guidance booking voice agent."""

# ruff: noqa: E501

from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

from pipecat.services.llm_service import FunctionCallParams

from campusx_voice_agent.config import DATA_DIR
from campusx_voice_agent.knowledge_base import search_knowledge_base

GUIDANCE_SYSTEM_PROMPT = """
# Personality

You are a friendly and helpful **CampusX support specialist**.

You know about CampusX courses, mentorship programs, YouTube content, CampusX One, notes, communities, and caller support.

You are patient and easy to talk to. Dont give answers longer than 2 to 3 sentences. Never overwhelm the callers.

Your main job is to help callers understand CampusX and find the right learning resource for their goals.

# Environment

You are talking to **CampusX callers and people who are interested in CampusX** through a voice conversation.

Callers may ask about:

- Courses and mentorship programs
- DSMP and DAMP
- CampusX One
- Course prices and access
- Course curriculum
- Prerequisites
- YouTube videos and notes
- AI and ML learning paths
- Doubt support and communities
- Refunds
- Career and interview preparation
- Technical learning resources

Many callers may be beginners and may not know which CampusX resource is right for them.

You have access to the **CampusX knowledge base**. Use it whenever you need current or specific CampusX information. Some runs may also inject relevant Moss semantic-retrieval passages directly into your context before you answer.

# Tone

You are patient and easy to talk to. Never overwhelm the callers.

Dont give answers longer than 2 to 3 sentences

If the caller speaks mainly in English, respond mainly in English.

Don't give answers having length of more than 2 sentences - unless the caller needs more detail.

# Goal

Your goal is to **help the caller get the correct information and choose the right CampusX resource**.

Follow this approach:

0. use 1 sentence at a time in your answer
1. First understand what the caller wants to achieve.
2. Understand their current level if it is relevant.
3. Use any injected Moss retrieval context first when it is relevant. If it is absent or insufficient, check the CampusX knowledge base tool when specific or current information is needed.
4. Give a simple and accurate answer.
5. If you recommend something, explain briefly why it is a good fit.
6. If you cannot verify the information, be honest and say so.
7. For account, payment, refund, or other issues that require human help, direct the caller to CampusX support.

**Important:** Never present a guess as a CampusX fact.

# Tools

name : search_campusx_knowledge_base, get_available_guidance_slots, book_free_guidance_session

Use injected Moss retrieval context when it directly answers the caller. Call `search_campusx_knowledge_base` whenever Moss context is absent, incomplete, or the caller asks about current or specific CampusX
information such as schedules, session names, timings, course details, or anything you cannot
answer confidently from memory. Search before ever saying "I'm not sure" or "I can't fetch that".
If the search returns no relevant result, say you could not verify it and direct the caller to
CampusX support.

Use `get_available_guidance_slots` and `book_free_guidance_session` only if a caller explicitly
asks you to book a free 20-minute guidance session.
Speak in short, warm, natural sentences without markdown.

Collect these details one at a time: full name, email address, learning or career goal,
and preferred day or time of day. Use the availability tool before offering slots.
Offer no more than three available slots at once.

Before booking, repeat the name, email, learning goal, and exact slot, then ask for an
explicit yes or no. Call the booking tool only after the caller says yes. Never invent
a slot or booking ID. If an email looks unclear, ask the caller to spell it.

# Guardrails

Never make up CampusX information.

Never promise:

- A job or placement
- A specific salary
- Future course launches
- Discounts or coupon codes
- A caller's account or enrollment status
- Refund approval
- Private community links
- Live session timings unless they are verified
- Instructor availability
- Information that is not published by CampusX

Do not say that CampusX guarantees a job or placement.

When talking about prices, course access, curriculum, schedules, or policies, always use the latest verified information available from injected Moss context or the knowledge base tool.

If you don't know something, simply say:

> "I'm not sure about that, and I don't want to give you the wrong information."

Then direct the caller to CampusX support if necessary.

For account-specific or unresolved issues, direct the caller to:

**support@campusx.in**

# Error Handling

If you cannot access or verify the required information, say:

> "I'm having trouble checking that information right now. I don't want to give you the wrong information."

Do not guess.

If the question is outside the verified CampusX information, say:

> "I don't want to guess here. Please check with the CampusX support team at support@campusx.in."

If a caller has an account-specific problem, payment issue, refund issue, or another problem that requires human help, direct them to CampusX support.

# Important Rule

**Your personality and behavior come from this system prompt.**

**Current CampusX information comes from injected Moss context when enabled, or from the knowledge base tool.**

Use the knowledge base to get updated information without changing your core personality, tone, or behavior.
""".strip()

_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SLOT_TIMES = (time(11, 0), time(16, 0), time(19, 0))

async def search_campusx_knowledge_base(
    params: FunctionCallParams,
    query: str,
):
    """Search verified CampusX knowledge-base files.

    Args:
        query: The caller's question or the CampusX topic to verify.
    """
    await params.result_callback(search_knowledge_base(query))

def _database_path() -> Path:
    configured = os.getenv("GUIDANCE_DB_PATH")
    return Path(configured) if configured else DATA_DIR / "guidance_bookings.db"


def _create_bookings_table(connection: sqlite3.Connection) -> None:
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


def _booking_columns(connection: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in connection.execute("PRAGMA table_info(bookings)")}


def _ensure_bookings_schema(connection: sqlite3.Connection) -> None:
    _create_bookings_table(connection)
    columns = _booking_columns(connection)

    required_columns = {"booking_id", "caller_name", "email", "learning_goal", "slot", "created_at"}
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise RuntimeError(f"Bookings database is missing required column(s): {missing}")

    connection.commit()


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _ensure_bookings_schema(connection)
    return connection


def _future_slots(days: int = 7) -> list[tuple[datetime, str]]:
    slots: list[tuple[datetime, str]] = []
    today = date.today()
    for day_offset in range(1, days + 1):
        slot_date = today + timedelta(days=day_offset)
        if slot_date.weekday() == 6:
            continue
        for slot_time in _SLOT_TIMES:
            value = datetime.combine(slot_date, slot_time)
            label = value.strftime("%A, %d %B %Y at %I:%M %p IST")
            slots.append((value, label))
    return slots


def available_guidance_slots(time_preference: str = "any", limit: int = 6) -> list[str]:
    """Return bookable slots, excluding reservations already stored in SQLite."""
    preference = time_preference.strip().lower()
    with _connect() as connection:
        booked = {row["slot"] for row in connection.execute("SELECT slot FROM bookings")}

    candidates = _future_slots()
    if "morning" in preference:
        candidates = [item for item in candidates if item[0].hour < 12]
    elif "afternoon" in preference:
        candidates = [item for item in candidates if 12 <= item[0].hour < 18]
    elif "evening" in preference:
        candidates = [item for item in candidates if item[0].hour >= 18]

    return [label for _, label in candidates if label not in booked][:limit]


async def get_available_guidance_slots(
    params: FunctionCallParams,
    time_preference: str,
):
    """Find currently available free-guidance slots.

    Args:
        time_preference: Caller preference such as morning, afternoon, evening, or any.
    """
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
    name = caller_name.strip()
    normalized_email = email.strip().lower().replace(" ", "")
    goal = learning_goal.strip()
    valid_slots = {label for _, label in _future_slots()}

    if len(name) < 2:
        result = {"booked": False, "error": "A full name is required."}
    elif not _EMAIL_PATTERN.match(normalized_email):
        result = {"booked": False, "error": "The email address is not valid."}
    elif len(goal) < 3:
        result = {"booked": False, "error": "A learning goal is required."}
    elif preferred_slot not in valid_slots:
        result = {"booked": False, "error": "Choose an exact slot returned by availability."}
    else:
        booking_id = f"CX-{uuid.uuid4().hex[:8].upper()}"
        try:
            with _connect() as connection:
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
            result = {
                "booked": False,
                "error": "That slot was just booked. Please check availability again.",
            }

    await params.result_callback(result)


def list_saved_bookings() -> list[dict[str, str]]:
    """Return bookings for the saved guidance-session viewer command."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT booking_id, caller_name, email, learning_goal, slot, created_at
            FROM bookings
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


GUIDANCE_TOOLS = [
    search_campusx_knowledge_base,
    get_available_guidance_slots,
    book_free_guidance_session,
]

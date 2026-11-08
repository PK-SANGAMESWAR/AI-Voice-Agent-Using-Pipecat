"""CLI rendering for guidance-session bookings.

Run with: uv run python -m campusx_voice_agent.bookings
"""

from __future__ import annotations

from campusx_voice_agent.tools import list_saved_bookings


def show_bookings() -> None:
    bookings = list_saved_bookings()
    if not bookings:
        print("No guidance sessions have been booked yet.")
        return

    print(f"Saved guidance bookings: {len(bookings)}\n")
    for booking in bookings:
        print(f"Booking ID : {booking['booking_id']}")
        print(f"Caller     : {booking['caller_name']}")
        print(f"Email      : {booking['email']}")
        print(f"Goal       : {booking['learning_goal']}")
        print(f"Slot       : {booking['slot']}")
        print(f"Created    : {booking['created_at']}")
        print("-" * 64)


if __name__ == "__main__":
    show_bookings()

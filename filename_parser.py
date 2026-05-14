#!/usr/bin/env python3
"""Parse Telegram audio filename metadata and normalize contact information."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TelegramAudioMetadata:
    """Parsed metadata from Telegram audio filename."""
    datetime_str: str  # "DD-MM-YYYY_HH-MM"
    call_id: str  # 10-digit phone (e.g., "0456307014")
    direction: str  # "incoming" or "outgoing"
    contact_id: Optional[str] = None  # 10-digit phone or None
    contact_name: Optional[str] = None  # Name from filename or None


def parse_telegram_filename(filename: str) -> Optional[TelegramAudioMetadata]:
    """
    Parse Telegram audio filename format:
    DD-MM-YYYY_HH-MM_CALLID_DIRECTION[_CONTACTID]_user_[CONTACTNAME].mp3

    Examples:
    - 12-10-2020_06-11_0456307014_incoming_0443334085_user_.mp3
    - 03-11-2020_09-39_0674840681_incoming_0503225972_user_Владислав Ярославцев.mp3
    - 28-12-2020_16-27_0675404242_outgoing__user_.mp3

    Returns None if filename does not match expected pattern.
    """
    # Remove extension
    base = filename
    for ext in [".mp3", ".m4a", ".wav"]:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    # Split by underscore
    parts = base.split("_")

    # Minimum: DATE TIME CALLID DIRECTION user [NAME...]
    if len(parts) < 5:
        return None

    try:
        # Extract date and time
        date_part = parts[0]  # DD-MM-YYYY
        time_part = parts[1]  # HH-MM
        datetime_str = f"{date_part}_{time_part}"

        # Validate datetime format
        _ = datetime.strptime(datetime_str, "%d-%m-%Y_%H-%M")

        # Extract call ID (originating phone)
        call_id = parts[2]
        if not re.match(r"^0\d{9}$", call_id):
            return None

        # Extract direction
        direction = parts[3]
        if direction not in ("incoming", "outgoing"):
            return None

        # Extract contact ID (if present) and find "user" marker
        contact_id = None
        contact_name = None
        user_idx = -1

        # Check if parts[4] is a phone number (contact ID)
        if len(parts) > 4 and re.match(r"^\d{10}$", parts[4]):
            contact_id = parts[4]
            search_start = 5
        else:
            search_start = 4

        # Find "user" marker
        for i in range(search_start, len(parts)):
            if "user" in parts[i]:
                user_idx = i
                break

        # Extract contact name after "user" marker
        if user_idx >= 0 and user_idx + 1 < len(parts):
            name_parts = parts[user_idx + 1 :]
            name_str = "_".join(name_parts).strip()
            if name_str:
                contact_name = unicodedata.normalize("NFC", name_str)

        return TelegramAudioMetadata(
            datetime_str=datetime_str,
            call_id=call_id,
            direction=direction,
            contact_id=contact_id,
            contact_name=contact_name,
        )

    except Exception:
        return None

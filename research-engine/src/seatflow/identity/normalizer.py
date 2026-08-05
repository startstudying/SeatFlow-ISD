"""Identity normalizer for raw seat and stock names."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedName:
    """Result of normalizing a raw name."""

    normalized: str
    original: str
    branch_suffix_removed: bool
    punctuation_normalized: bool


def normalize_seat_name(raw: str) -> NormalizedName:
    """Normalize a seat name for matching.

    Steps:
    1. Strip whitespace.
    2. Remove branch suffixes like '#1', '(第1营业部)', '（分公司）'.
    3. Normalize full-width punctuation to half-width.
    4. Collapse internal whitespace.
    """
    original = raw
    s = raw.strip()
    # Remove branch suffixes
    suffix_pattern = r"[#（(].*?$"
    s_before = s
    s = re.sub(suffix_pattern, "", s).strip()
    branch_removed = s != s_before
    # Normalize full-width to half-width
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("。", ".")
    s = s.replace("：", ":").replace("；", ";")
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    return NormalizedName(
        normalized=s,
        original=original,
        branch_suffix_removed=branch_removed,
        punctuation_normalized=True,
    )


def normalize_stock_code(raw: str) -> str:
    """Normalize a stock code to TS_CODE format (e.g. 000001.SZ)."""
    s = raw.strip().upper()
    # Remove spaces
    s = re.sub(r"\s+", "", s)
    # If already has exchange suffix
    if "." in s:
        return s
    # Infer exchange from code prefix
    if s.startswith(("60", "68", "90", "11", "13", "56")):
        return f"{s}.SH"
    if s.startswith(("00", "30", "12", "15", "159")):
        return f"{s}.SZ"
    if s.startswith(("43", "83", "87", "88")):
        return f"{s}.BJ"
    return s

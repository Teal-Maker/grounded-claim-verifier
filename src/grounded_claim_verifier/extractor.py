"""Numeric claim extraction from free-form text.

Extracts typed measurements (temperature and pressure) from LLM-generated
text using regular expressions.  Range expressions in source material are
also captured so the verifier can check containment.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Temperature: optional sign + digits followed by °F or °C (with optional
# space).  Negative lookbehind prevents matching range separators, e.g.
# "53-59°F" would otherwise yield -59 as well as 59.
TEMP_RE = re.compile(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*°\s*([FC])", re.IGNORECASE)

# Pressure: optional sign + digits followed by psi/psig/kPa/bar.
PRESSURE_RE = re.compile(
    r"(?<!\d)(-?\d+(?:\.\d+)?)\s*(psi|psig|kpa|bar)\b", re.IGNORECASE
)

# Range patterns in source text: "X-Y unit" (en-dash / em-dash variants).
TEMP_RANGE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(-?\d+(?:\.\d+)?)\s*°\s*([FC])",
    re.IGNORECASE,
)
PRESSURE_RANGE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(-?\d+(?:\.\d+)?)\s*(psi|psig|kpa|bar)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Unit family mapping
# ---------------------------------------------------------------------------

TEMP_UNITS: set[str] = {"°F", "°C"}
PRESSURE_UNITS: set[str] = {"PSI", "PSIG", "KPA", "BAR"}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _unit_family(unit: str) -> str:
    """Return ``'temperature'`` or ``'pressure'`` for a unit string.

    Parameters
    ----------
    unit:
        Raw unit string as it appears in the claim or source text,
        e.g. ``"°F"``, ``"PSIG"``, ``"kPa"``.

    Returns
    -------
    str
        ``'temperature'``, ``'pressure'``, or ``'unknown'``.
    """
    norm = unit.upper().lstrip("°")
    if norm in ("F", "C") or f"°{norm}" in TEMP_UNITS:
        return "temperature"
    if norm in PRESSURE_UNITS:
        return "pressure"
    return "unknown"


def _source_context(text: str, start: int, end: int, window: int = 80) -> str:
    """Return the text window surrounding a regex match position.

    Parameters
    ----------
    text:
        Full source string.
    start:
        Match start index.
    end:
        Match end index.
    window:
        Number of characters to include on each side (default: 80).

    Returns
    -------
    str
        Stripped context string with newlines collapsed to spaces.
    """
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end].replace("\n", " ").strip()


def _extract_typed_measurements(text: str) -> list[dict[str, Any]]:
    """Extract typed measurements (value + unit + family) from source text.

    Parameters
    ----------
    text:
        Source text to scan.

    Returns
    -------
    list[dict]
        Each element has keys ``value``, ``unit``, ``family``, ``start``,
        ``end``.
    """
    results: list[dict[str, Any]] = []

    for m in TEMP_RE.finditer(text):
        results.append({
            "value": float(m.group(1)),
            "unit": "°" + m.group(2).upper(),
            "family": "temperature",
            "start": m.start(),
            "end": m.end(),
        })

    for m in PRESSURE_RE.finditer(text):
        results.append({
            "value": float(m.group(1)),
            "unit": m.group(2).upper(),
            "family": "pressure",
            "start": m.start(),
            "end": m.end(),
        })

    return results


def _extract_ranges(text: str) -> list[dict[str, Any]]:
    """Extract range expressions (``X-Y unit``) from source text.

    Low and high values are normalised so that ``low <= high`` regardless
    of the order they appear in the source.

    Parameters
    ----------
    text:
        Source text to scan.

    Returns
    -------
    list[dict]
        Each element has keys ``low``, ``high``, ``family``, ``start``,
        ``end``.
    """
    results: list[dict[str, Any]] = []

    for m in TEMP_RANGE_RE.finditer(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        results.append({
            "low": lo,
            "high": hi,
            "family": "temperature",
            "start": m.start(),
            "end": m.end(),
        })

    for m in PRESSURE_RANGE_RE.finditer(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        results.append({
            "low": lo,
            "high": hi,
            "family": "pressure",
            "start": m.start(),
            "end": m.end(),
        })

    return results


def extract_claims(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract all temperature and pressure claims from a list of QA records.

    Supports two record shapes:

    * **Single-turn**: ``{"qa_id": ..., "source_content_id": ..., "answer": "..."}``
    * **Multi-turn**: ``{"type": "multi_turn", "conversation": [{"role": "assistant", "content": "..."}]}``

    Parameters
    ----------
    records:
        List of QA record dicts as loaded from a JSONL dataset file.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(temp_claims, pressure_claims)``.  Each claim dict contains:

        * ``qa_id`` — identifier of the QA pair
        * ``source_content_id`` — identifier of the originating source document
        * ``value`` — numeric value as ``float``
        * ``unit`` — normalised unit string (e.g. ``"°F"``, ``"PSIG"``)
        * ``claim_text`` — the matched substring
        * ``context`` — up to 80 chars of surrounding text on each side
        * ``turn_index`` — 0 for single-turn; conversation index for multi-turn
    """
    temp_claims: list[dict[str, Any]] = []
    pressure_claims: list[dict[str, Any]] = []

    for rec in records:
        qa_id = rec.get("qa_id", "")
        src_id = rec.get("source_content_id", "")

        # Collect (index, text) pairs for assistant turns.
        if rec.get("type") == "multi_turn":
            turns: list[tuple[int, str]] = [
                (i, t["content"])
                for i, t in enumerate(rec.get("conversation", []))
                if t.get("role") == "assistant"
            ]
        else:
            turns = [(0, rec.get("answer", ""))]

        for turn_idx, text in turns:
            # Temperature claims
            for m in TEMP_RE.finditer(text):
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                temp_claims.append({
                    "qa_id": qa_id,
                    "source_content_id": src_id,
                    "value": float(m.group(1)),
                    "unit": "°" + m.group(2).upper(),
                    "claim_text": m.group(0),
                    "context": text[start:end].replace("\n", " ").strip(),
                    "turn_index": turn_idx,
                })

            # Pressure claims
            for m in PRESSURE_RE.finditer(text):
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                pressure_claims.append({
                    "qa_id": qa_id,
                    "source_content_id": src_id,
                    "value": float(m.group(1)),
                    "unit": m.group(2).upper(),
                    "claim_text": m.group(0),
                    "context": text[start:end].replace("\n", " ").strip(),
                    "turn_index": turn_idx,
                })

    return temp_claims, pressure_claims

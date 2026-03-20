"""Source-grounded claim verification.

The verifier applies four matching layers in priority order:

1. **Exact match** — the source text contains the same numeric value with a
   unit of the same family (temperature or pressure).
2. **Approximate match** — the source value is within a configurable
   tolerance (default ±10 %).  Negative values are handled by swapping
   the bound directions.
3. **Range match** — the claimed value falls within a range expression
   found in the source (e.g. ``100-200 psig``).
4. **Untyped fallback** — bare digits appear in the source text and the
   surrounding context (~30 chars each side) contains a measurement-adjacent
   keyword for the same unit family (handles transcripts where unit symbols
   are often dropped).

Possible verdicts:
* ``confirmed``         — exact typed match
* ``approximate``       — within tolerance
* ``in_range``          — contained in a source range
* ``confirmed_untyped`` — exact bare number with context keyword
* ``not_found``         — none of the above matched
* ``no_source``         — no source text was available for this claim
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from grounded_claim_verifier.extractor import (
    _extract_ranges,
    _extract_typed_measurements,
    _source_context,
    _unit_family,
)
from grounded_claim_verifier.providers.base import SourceProvider

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE = 0.10

# ---------------------------------------------------------------------------
# Default context-keyword patterns
# ---------------------------------------------------------------------------

_DEFAULT_TEMP_CONTEXT = re.compile(
    r"(?:degree|temp|°|fahrenheit|celsius|superheat|subcool|saturat)",
    re.IGNORECASE,
)
_DEFAULT_PRESSURE_CONTEXT = re.compile(
    r"(?:psi|psig|pressure|head|suction|discharg|bar\b|kpa)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VerifierConfig:
    """Configuration for :class:`ClaimVerifier`.

    Parameters
    ----------
    tolerance:
        Fractional tolerance for approximate matching (default: ``0.10``
        meaning ±10 %).
    temp_context_keywords:
        Compiled regex used in the untyped fallback for temperature claims.
        Defaults to a built-in pattern covering common temperature vocabulary.
    pressure_context_keywords:
        Compiled regex used in the untyped fallback for pressure claims.
        Defaults to a built-in pattern covering common pressure vocabulary.
    """

    tolerance: float = DEFAULT_TOLERANCE
    temp_context_keywords: re.Pattern[str] | None = field(default=None)
    pressure_context_keywords: re.Pattern[str] | None = field(default=None)

    def __post_init__(self) -> None:
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(
                f"tolerance must be between 0.0 and 1.0, got {self.tolerance}"
            )
        if self.temp_context_keywords is None:
            self.temp_context_keywords = _DEFAULT_TEMP_CONTEXT
        if self.pressure_context_keywords is None:
            self.pressure_context_keywords = _DEFAULT_PRESSURE_CONTEXT


# ---------------------------------------------------------------------------
# Standalone function (mirrors original API)
# ---------------------------------------------------------------------------


def verify_claim(
    claim: dict[str, Any],
    source_text: str | None,
    tolerance: float = DEFAULT_TOLERANCE,
    temp_context_keywords: re.Pattern[str] | None = None,
    pressure_context_keywords: re.Pattern[str] | None = None,
) -> dict[str, Any]:
    """Verify a single claim against source text using unit-aware matching.

    This is the low-level function that implements all four matching layers.
    It is also called internally by :class:`ClaimVerifier`.

    Parameters
    ----------
    claim:
        Claim dict produced by :func:`~grounded_claim_verifier.extractor.extract_claims`
        (must contain at minimum ``value`` and ``unit`` keys).
    source_text:
        The originating source text, or ``None`` if unavailable.
    tolerance:
        Fractional tolerance for approximate matching (default: 0.10).
    temp_context_keywords:
        Override for temperature context keywords regex.
    pressure_context_keywords:
        Override for pressure context keywords regex.

    Returns
    -------
    dict
        The original claim dict extended with ``verdict``, ``source_match``,
        ``source_context``, and ``match_type`` keys.
    """
    if not (0.0 <= tolerance <= 1.0):
        raise ValueError(
            f"tolerance must be between 0.0 and 1.0, got {tolerance}"
        )
    if "value" not in claim:
        raise ValueError(f"Claim dict missing required 'value' key: {claim!r}")

    _temp_ctx = temp_context_keywords or _DEFAULT_TEMP_CONTEXT
    _pressure_ctx = pressure_context_keywords or _DEFAULT_PRESSURE_CONTEXT

    if source_text is None:
        return {
            **claim,
            "verdict": "no_source",
            "source_match": None,
            "source_context": None,
            "match_type": None,
        }

    target_value = claim["value"]
    claim_family = _unit_family(claim.get("unit", ""))

    # Extract typed measurements from source (same unit family only).
    source_measurements = [
        m
        for m in _extract_typed_measurements(source_text)
        if m["family"] == claim_family
    ]

    # ------------------------------------------------------------------
    # 1. Exact match (same family)
    # ------------------------------------------------------------------
    for sm in source_measurements:
        if sm["value"] == target_value:
            return {
                **claim,
                "verdict": "confirmed",
                "source_match": sm["value"],
                "source_context": _source_context(source_text, sm["start"], sm["end"]),
                "match_type": "exact",
            }

    # ------------------------------------------------------------------
    # 2. Approximate match within tolerance (same family)
    # ------------------------------------------------------------------
    if target_value != 0:
        if target_value > 0:
            lower = target_value * (1 - tolerance)
            upper = target_value * (1 + tolerance)
        else:
            # For negative values, swap bounds so lower < upper.
            lower = target_value * (1 + tolerance)
            upper = target_value * (1 - tolerance)

        best_match: dict[str, Any] | None = None
        best_distance = float("inf")

        for sm in source_measurements:
            if lower <= sm["value"] <= upper:
                distance = abs(sm["value"] - target_value)
                if distance < best_distance:
                    best_distance = distance
                    best_match = sm

        if best_match is not None:
            return {
                **claim,
                "verdict": "approximate",
                "source_match": best_match["value"],
                "source_context": _source_context(
                    source_text, best_match["start"], best_match["end"]
                ),
                "match_type": f"within_{tolerance:.0%}",
            }

    # ------------------------------------------------------------------
    # 3. Range match: check if claim falls within a source range
    # ------------------------------------------------------------------
    source_ranges = [
        r for r in _extract_ranges(source_text) if r["family"] == claim_family
    ]
    for rng in source_ranges:
        if rng["low"] <= target_value <= rng["high"]:
            return {
                **claim,
                "verdict": "in_range",
                "source_match": f"{rng['low']}-{rng['high']}",
                "source_context": _source_context(
                    source_text, rng["start"], rng["end"]
                ),
                "match_type": "within_source_range",
            }

    # ------------------------------------------------------------------
    # 4. Untyped fallback: bare number + measurement-adjacent keyword
    #    in ±30 char window.  Handles transcripts that drop unit symbols.
    # ------------------------------------------------------------------
    if claim_family == "unknown":
        # Skip untyped fallback for unsupported units — there is no
        # meaningful context keyword set to apply, and defaulting to the
        # pressure keywords would silently produce false positives.
        return {
            **claim,
            "verdict": "not_found",
            "source_match": None,
            "source_context": None,
            "match_type": None,
        }

    context_re = _temp_ctx if claim_family == "temperature" else _pressure_ctx

    for m in re.finditer(r"-?\d+(?:\.\d+)?", source_text):
        try:
            val = float(m.group())
        except ValueError:
            continue
        if val != target_value:
            continue
        ctx_start = max(0, m.start() - 30)
        ctx_end = min(len(source_text), m.end() + 30)
        nearby = source_text[ctx_start:ctx_end]
        if context_re.search(nearby):
            return {
                **claim,
                "verdict": "confirmed_untyped",
                "source_match": val,
                "source_context": _source_context(source_text, m.start(), m.end()),
                "match_type": "exact_untyped_with_context",
            }

    # ------------------------------------------------------------------
    # Not found in source
    # ------------------------------------------------------------------
    return {
        **claim,
        "verdict": "not_found",
        "source_match": None,
        "source_context": None,
        "match_type": None,
    }


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------


class ClaimVerifier:
    """Verify a batch of claims against source texts fetched via a provider.

    Parameters
    ----------
    source_provider:
        Any object implementing the :class:`~grounded_claim_verifier.providers.SourceProvider`
        protocol.  Used to retrieve the originating source text for each claim.
    config:
        Optional :class:`VerifierConfig`.  Defaults are used when omitted.

    Examples
    --------
    >>> from grounded_claim_verifier import ClaimVerifier, JSONLProvider
    >>> provider = JSONLProvider("sources.jsonl")
    >>> verifier = ClaimVerifier(provider)
    >>> results = verifier.verify_claims(claims)
    """

    def __init__(
        self,
        source_provider: SourceProvider,
        config: VerifierConfig | None = None,
    ) -> None:
        self._provider = source_provider
        self._config = config or VerifierConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_claims(self, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Verify all claims against their source texts.

        Source texts are fetched in a single batch call to the provider so
        that database-backed providers can use a single round-trip.

        Parameters
        ----------
        claims:
            List of claim dicts as produced by
            :func:`~grounded_claim_verifier.extractor.extract_claims` (must
            contain a ``source_content_id`` key).

        Returns
        -------
        list[dict]
            Each input claim dict extended with verification verdict fields.
        """
        # Batch-fetch all required source texts.
        source_ids = [
            str(c["source_content_id"])
            for c in claims
            if c.get("source_content_id") is not None
        ]
        unique_ids = list(dict.fromkeys(source_ids))  # preserve order, deduplicate
        source_texts = self._provider.fetch_texts(unique_ids)

        results: list[dict[str, Any]] = []
        for claim in claims:
            sid = str(claim.get("source_content_id", ""))
            src_text = source_texts.get(sid)
            results.append(self.verify_single(claim, src_text))

        return results

    def verify_single(
        self,
        claim: dict[str, Any],
        source_text: str | None,
    ) -> dict[str, Any]:
        """Verify a single claim against an already-retrieved source text.

        Use this method when you have already fetched the source text, e.g.
        when iterating over claims one by one or in tests.

        Parameters
        ----------
        claim:
            Claim dict (must contain ``value`` and ``unit`` keys).
        source_text:
            The originating source text, or ``None`` if unavailable.

        Returns
        -------
        dict
            Claim dict extended with verdict fields.
        """
        cfg = self._config
        return verify_claim(
            claim,
            source_text,
            tolerance=cfg.tolerance,
            temp_context_keywords=cfg.temp_context_keywords,
            pressure_context_keywords=cfg.pressure_context_keywords,
        )

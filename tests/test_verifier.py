"""Tests for grounded_claim_verifier.verifier."""

from __future__ import annotations

import re

import pytest

from grounded_claim_verifier.verifier import (
    ClaimVerifier,
    VerifierConfig,
    verify_claim,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _claim(
    value: float,
    unit: str = "°F",
    qa_id: str = "qa-test",
    source_content_id: str = "src-test",
) -> dict:
    return {
        "qa_id": qa_id,
        "source_content_id": source_content_id,
        "value": value,
        "unit": unit,
        "claim_text": f"{value}{unit}",
        "context": "",
        "turn_index": 0,
    }


# ---------------------------------------------------------------------------
# verify_claim — no_source
# ---------------------------------------------------------------------------


class TestVerifyClaimNoSource:
    def test_none_source_gives_no_source_verdict(self) -> None:
        result = verify_claim(_claim(100.0), source_text=None)
        assert result["verdict"] == "no_source"
        assert result["source_match"] is None
        assert result["match_type"] is None

    def test_original_claim_fields_preserved(self) -> None:
        claim = _claim(100.0)
        result = verify_claim(claim, source_text=None)
        assert result["qa_id"] == claim["qa_id"]
        assert result["value"] == 100.0


# ---------------------------------------------------------------------------
# verify_claim — exact match
# ---------------------------------------------------------------------------


class TestVerifyClaimExact:
    def test_exact_temperature(self) -> None:
        result = verify_claim(_claim(125.0, "°F"), source_text="Discharge temp is 125°F.")
        assert result["verdict"] == "confirmed"
        assert result["match_type"] == "exact"
        assert result["source_match"] == 125.0

    def test_exact_pressure(self) -> None:
        result = verify_claim(_claim(70.0, "PSIG"), source_text="Suction at 70 psig.")
        assert result["verdict"] == "confirmed"

    def test_exact_kpa(self) -> None:
        result = verify_claim(_claim(150.0, "KPA"), source_text="System pressure 150 kPa.")
        assert result["verdict"] == "confirmed"

    def test_negative_exact(self) -> None:
        result = verify_claim(_claim(-10.0, "°C"), source_text="Ambient was -10°C.")
        assert result["verdict"] == "confirmed"

    def test_source_context_populated(self) -> None:
        result = verify_claim(_claim(125.0, "°F"), source_text="Discharge temp is 125°F.")
        assert result["source_context"] is not None
        assert "125" in result["source_context"]


# ---------------------------------------------------------------------------
# verify_claim — approximate match
# ---------------------------------------------------------------------------


class TestVerifyClaimApproximate:
    def test_within_10_percent(self) -> None:
        # 130 is within 10% of 125 (range 112.5 to 137.5)
        result = verify_claim(_claim(125.0, "°F"), source_text="Temp is 130°F.", tolerance=0.10)
        assert result["verdict"] == "approximate"
        assert result["match_type"] == "within_10%"

    def test_just_outside_tolerance(self) -> None:
        # 140 > 137.5 (10% above 125)
        result = verify_claim(_claim(125.0, "°F"), source_text="Temp is 140°F.", tolerance=0.10)
        assert result["verdict"] != "approximate"

    def test_negative_value_approximate(self) -> None:
        # -11 is within 10% of -10 (bounds: -11 to -9 after swap)
        result = verify_claim(_claim(-10.0, "°C"), source_text="Temperature -11°C.", tolerance=0.10)
        assert result["verdict"] == "approximate"

    def test_zero_value_skips_approximate(self) -> None:
        # value == 0: approximate check is skipped (division by zero guard)
        result = verify_claim(_claim(0.0, "°F"), source_text="Temperature 0°F.")
        # Exact match should fire instead
        assert result["verdict"] == "confirmed"

    def test_custom_tolerance(self) -> None:
        # With 5% tolerance, 130 should NOT match 125 (range 118.75 to 131.25)
        result = verify_claim(_claim(125.0, "°F"), source_text="Temp is 130°F.", tolerance=0.05)
        # 130 <= 131.25, so still in range — let's use a value clearly outside
        result2 = verify_claim(_claim(125.0, "°F"), source_text="Temp is 133°F.", tolerance=0.05)
        assert result2["verdict"] not in ("confirmed", "approximate")

    def test_best_match_is_closest(self) -> None:
        # Source has two values within tolerance; the closer one should win.
        result = verify_claim(
            _claim(100.0, "°F"),
            source_text="Could be 102°F or 108°F.",
            tolerance=0.10,
        )
        assert result["verdict"] == "approximate"
        assert result["source_match"] == 102.0


# ---------------------------------------------------------------------------
# verify_claim — range match
# ---------------------------------------------------------------------------


class TestVerifyClaimRange:
    def test_within_range(self) -> None:
        # Use a value mid-range (75) that does not sit within 10% of either
        # endpoint (60 or 80), so the approximate layer does not fire first.
        result = verify_claim(
            _claim(75.0, "PSIG"),
            source_text="Normal suction pressure is 50-100 psig.",
        )
        assert result["verdict"] == "in_range"
        assert result["match_type"] == "within_source_range"

    def test_at_range_boundary(self) -> None:
        result = verify_claim(
            _claim(60.0, "PSIG"),
            source_text="Normal range is 60-80 psig.",
        )
        assert result["verdict"] in ("confirmed", "in_range")

    def test_outside_range(self) -> None:
        result = verify_claim(
            _claim(99.0, "PSIG"),
            source_text="Normal range is 60-80 psig.",
        )
        assert result["verdict"] == "not_found"

    def test_temp_range(self) -> None:
        # 125°F is within 10% of 120 (range 108-132), so use a wide range
        # with endpoints far enough away that approximate doesn't fire.
        result = verify_claim(
            _claim(125.0, "°F"),
            source_text="Discharge range is 100-150°F.",
        )
        assert result["verdict"] == "in_range"

    def test_range_family_isolation(self) -> None:
        # A pressure claim should not match a temperature range.
        result = verify_claim(
            _claim(75.0, "PSIG"),
            source_text="Normal temps are 60-80°F.",
        )
        assert result["verdict"] == "not_found"


# ---------------------------------------------------------------------------
# verify_claim — untyped fallback
# ---------------------------------------------------------------------------


class TestVerifyClaimUntyped:
    def test_temp_untyped_fallback(self) -> None:
        # Source has bare number with temperature context keyword.
        result = verify_claim(
            _claim(65.0, "°F"),
            source_text="The superheat should be 65 at those conditions.",
        )
        assert result["verdict"] == "confirmed_untyped"
        assert result["match_type"] == "exact_untyped_with_context"

    def test_pressure_untyped_fallback(self) -> None:
        result = verify_claim(
            _claim(70.0, "PSIG"),
            source_text="The suction was reading 70 at the gauge.",
        )
        assert result["verdict"] == "confirmed_untyped"

    def test_no_context_keyword_does_not_match(self) -> None:
        # Bare number present but no measurement context keyword nearby.
        result = verify_claim(
            _claim(65.0, "°F"),
            source_text="There were 65 technicians at the conference.",
        )
        assert result["verdict"] == "not_found"

    def test_custom_context_regex(self) -> None:
        custom_temp = re.compile(r"klima", re.IGNORECASE)
        result = verify_claim(
            _claim(25.0, "°C"),
            source_text="Klima reading was 25 in the room.",
            temp_context_keywords=custom_temp,
        )
        assert result["verdict"] == "confirmed_untyped"


# ---------------------------------------------------------------------------
# verify_claim — not_found
# ---------------------------------------------------------------------------


class TestVerifyClaimNotFound:
    def test_completely_wrong_value(self) -> None:
        result = verify_claim(_claim(999.0, "°F"), source_text="System was at 125°F.")
        assert result["verdict"] == "not_found"

    def test_wrong_family_not_matched(self) -> None:
        # Claim is temperature, source has matching number as pressure only.
        result = verify_claim(
            _claim(70.0, "°F"),
            source_text="The pressure is 70 psig.",
        )
        assert result["verdict"] == "not_found"


# ---------------------------------------------------------------------------
# VerifierConfig
# ---------------------------------------------------------------------------


class TestVerifierConfig:
    def test_defaults(self) -> None:
        cfg = VerifierConfig()
        assert cfg.tolerance == pytest.approx(0.10)
        assert cfg.temp_context_keywords is not None
        assert cfg.pressure_context_keywords is not None

    def test_custom_tolerance(self) -> None:
        cfg = VerifierConfig(tolerance=0.05)
        assert cfg.tolerance == pytest.approx(0.05)

    def test_custom_context_keywords(self) -> None:
        custom = re.compile(r"custom", re.IGNORECASE)
        cfg = VerifierConfig(temp_context_keywords=custom)
        assert cfg.temp_context_keywords is custom


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------


class DictProvider:
    """In-memory provider for testing."""

    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
        return {sid: self._data[sid] for sid in source_ids if sid in self._data}


class TestClaimVerifier:
    def test_verify_claims_basic(self) -> None:
        provider = DictProvider({"s1": "Discharge temp is 125°F."})
        verifier = ClaimVerifier(provider)
        claims = [_claim(125.0, "°F", source_content_id="s1")]
        results = verifier.verify_claims(claims)
        assert len(results) == 1
        assert results[0]["verdict"] == "confirmed"

    def test_verify_claims_missing_source(self) -> None:
        provider = DictProvider({})
        verifier = ClaimVerifier(provider)
        claims = [_claim(125.0, "°F", source_content_id="s-missing")]
        results = verifier.verify_claims(claims)
        assert results[0]["verdict"] == "no_source"

    def test_verify_claims_batch_deduplicates_ids(self, monkeypatch) -> None:
        call_log: list[list[str]] = []

        class TrackingProvider:
            def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
                call_log.append(list(source_ids))
                return {"s1": "125°F"}

        verifier = ClaimVerifier(TrackingProvider())
        claims = [
            _claim(125.0, "°F", source_content_id="s1"),
            _claim(125.0, "°F", source_content_id="s1"),
        ]
        verifier.verify_claims(claims)
        # fetch_texts should be called once; s1 should appear once in the IDs.
        assert len(call_log) == 1
        assert call_log[0].count("s1") == 1

    def test_verify_single(self) -> None:
        provider = DictProvider({"s1": "Pressure is 70 psig."})
        verifier = ClaimVerifier(provider)
        claim = _claim(70.0, "PSIG", source_content_id="s1")
        result = verifier.verify_single(claim, "Pressure is 70 psig.")
        assert result["verdict"] == "confirmed"

    def test_verify_single_no_source(self) -> None:
        provider = DictProvider({})
        verifier = ClaimVerifier(provider)
        result = verifier.verify_single(_claim(70.0, "PSIG"), None)
        assert result["verdict"] == "no_source"

    def test_config_tolerance_passed_through(self) -> None:
        provider = DictProvider({"s1": "Temp is 130°F."})
        config = VerifierConfig(tolerance=0.01)  # Very tight — 130 outside ±1% of 125
        verifier = ClaimVerifier(provider, config)
        claims = [_claim(125.0, "°F", source_content_id="s1")]
        results = verifier.verify_claims(claims)
        assert results[0]["verdict"] == "not_found"

    def test_multiple_claims_multiple_sources(self) -> None:
        provider = DictProvider({
            "s1": "Suction pressure 70 psig.",
            "s2": "Discharge temp 200°F.",
        })
        verifier = ClaimVerifier(provider)
        claims = [
            _claim(70.0, "PSIG", source_content_id="s1"),
            _claim(200.0, "°F", source_content_id="s2"),
            _claim(999.0, "°F", source_content_id="s1"),
        ]
        results = verifier.verify_claims(claims)
        assert results[0]["verdict"] == "confirmed"
        assert results[1]["verdict"] == "confirmed"
        assert results[2]["verdict"] == "not_found"

    def test_implements_protocol(self) -> None:
        """ClaimVerifier accepts any SourceProvider-conformant object."""
        from grounded_claim_verifier.providers.base import SourceProvider

        provider = DictProvider({})
        assert isinstance(provider, SourceProvider)

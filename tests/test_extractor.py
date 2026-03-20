"""Tests for grounded_claim_verifier.extractor."""

from __future__ import annotations

import pytest

from grounded_claim_verifier.extractor import (
    PRESSURE_RE,
    PRESSURE_RANGE_RE,
    TEMP_RE,
    TEMP_RANGE_RE,
    _extract_ranges,
    _extract_typed_measurements,
    _source_context,
    _unit_family,
    extract_claims,
)


# ---------------------------------------------------------------------------
# _unit_family
# ---------------------------------------------------------------------------


class TestUnitFamily:
    def test_fahrenheit_symbol(self) -> None:
        assert _unit_family("°F") == "temperature"

    def test_celsius_symbol(self) -> None:
        assert _unit_family("°C") == "temperature"

    def test_fahrenheit_no_degree(self) -> None:
        # As produced by extractor: "°" + "F"
        assert _unit_family("F") == "temperature"

    def test_celsius_no_degree(self) -> None:
        assert _unit_family("C") == "temperature"

    def test_psi(self) -> None:
        assert _unit_family("PSI") == "pressure"

    def test_psig(self) -> None:
        assert _unit_family("PSIG") == "pressure"

    def test_kpa(self) -> None:
        assert _unit_family("KPA") == "pressure"

    def test_bar(self) -> None:
        assert _unit_family("BAR") == "pressure"

    def test_case_insensitive(self) -> None:
        assert _unit_family("psig") == "pressure"

    def test_unknown_unit(self) -> None:
        assert _unit_family("XYZ") == "unknown"

    def test_empty_string(self) -> None:
        assert _unit_family("") == "unknown"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


class TestTempRegex:
    def test_simple_fahrenheit(self) -> None:
        m = TEMP_RE.search("125°F")
        assert m is not None
        assert float(m.group(1)) == 125.0
        assert m.group(2).upper() == "F"

    def test_simple_celsius(self) -> None:
        m = TEMP_RE.search("40°C")
        assert m is not None
        assert float(m.group(1)) == 40.0

    def test_space_before_unit(self) -> None:
        m = TEMP_RE.search("125 °F")
        assert m is not None

    def test_negative_temperature(self) -> None:
        m = TEMP_RE.search("-10°C")
        assert m is not None
        assert float(m.group(1)) == -10.0

    def test_decimal_temperature(self) -> None:
        m = TEMP_RE.search("98.6°F")
        assert m is not None
        assert float(m.group(1)) == pytest.approx(98.6)

    def test_range_does_not_produce_negative(self) -> None:
        # "53-59°F" — lookbehind should prevent -59 from matching.
        matches = TEMP_RE.findall("53-59°F")
        values = [float(m[0]) for m in matches]
        assert -59.0 not in values

    def test_multiple_matches(self) -> None:
        text = "Supply at 55°F, return at 75°F."
        matches = TEMP_RE.findall(text)
        assert len(matches) == 2


class TestPressureRegex:
    def test_psig(self) -> None:
        m = PRESSURE_RE.search("70 psig")
        assert m is not None
        assert float(m.group(1)) == 70.0

    def test_psi(self) -> None:
        m = PRESSURE_RE.search("150psi")
        assert m is not None
        assert float(m.group(1)) == 150.0

    def test_kpa(self) -> None:
        m = PRESSURE_RE.search("200 kPa")
        assert m is not None
        assert float(m.group(1)) == 200.0

    def test_bar(self) -> None:
        m = PRESSURE_RE.search("2.5 bar")
        assert m is not None
        assert float(m.group(1)) == pytest.approx(2.5)

    def test_negative_pressure(self) -> None:
        m = PRESSURE_RE.search("-5 psig")
        assert m is not None
        assert float(m.group(1)) == -5.0

    def test_no_match_partial_word(self) -> None:
        # "barrel" should not match "bar"
        m = PRESSURE_RE.search("100 barrel")
        assert m is None


class TestRangeRegex:
    def test_temp_range_hyphen(self) -> None:
        m = TEMP_RANGE_RE.search("120-130°F")
        assert m is not None
        assert float(m.group(1)) == 120.0
        assert float(m.group(2)) == 130.0

    def test_temp_range_en_dash(self) -> None:
        m = TEMP_RANGE_RE.search("120\u2013130°F")
        assert m is not None

    def test_pressure_range(self) -> None:
        m = PRESSURE_RANGE_RE.search("60-80 psig")
        assert m is not None
        assert float(m.group(1)) == 60.0
        assert float(m.group(2)) == 80.0


# ---------------------------------------------------------------------------
# _extract_typed_measurements
# ---------------------------------------------------------------------------


class TestExtractTypedMeasurements:
    def test_single_temp(self) -> None:
        results = _extract_typed_measurements("Discharge temp: 125°F")
        assert len(results) == 1
        r = results[0]
        assert r["value"] == 125.0
        assert r["family"] == "temperature"

    def test_single_pressure(self) -> None:
        results = _extract_typed_measurements("Suction at 70 psig")
        assert len(results) == 1
        assert results[0]["family"] == "pressure"

    def test_mixed(self) -> None:
        results = _extract_typed_measurements("125°F and 70 psig")
        families = {r["family"] for r in results}
        assert families == {"temperature", "pressure"}

    def test_empty_text(self) -> None:
        assert _extract_typed_measurements("") == []

    def test_no_measurements(self) -> None:
        assert _extract_typed_measurements("no numbers here") == []

    def test_start_end_positions(self) -> None:
        text = "temp is 125°F today"
        results = _extract_typed_measurements(text)
        assert len(results) == 1
        r = results[0]
        # The matched text at [start:end] should be "125°F"
        assert text[r["start"] : r["end"]] == "125°F"


# ---------------------------------------------------------------------------
# _extract_ranges
# ---------------------------------------------------------------------------


class TestExtractRanges:
    def test_temp_range(self) -> None:
        results = _extract_ranges("Normal: 120-130°F")
        assert len(results) == 1
        r = results[0]
        assert r["low"] == 120.0
        assert r["high"] == 130.0
        assert r["family"] == "temperature"

    def test_pressure_range(self) -> None:
        results = _extract_ranges("60-80 psig normal")
        assert len(results) == 1
        assert results[0]["family"] == "pressure"

    def test_inverted_range_is_normalised(self) -> None:
        # Source text with high-low ordering
        results = _extract_ranges("130-120°F")
        assert results[0]["low"] == 120.0
        assert results[0]["high"] == 130.0

    def test_no_ranges(self) -> None:
        assert _extract_ranges("pressure is 70 psig") == []


# ---------------------------------------------------------------------------
# _source_context
# ---------------------------------------------------------------------------


class TestSourceContext:
    def test_short_text(self) -> None:
        text = "abc"
        result = _source_context(text, 1, 2)
        assert result == "abc"

    def test_clamps_at_boundaries(self) -> None:
        text = "x" * 200
        result = _source_context(text, 0, 5)
        assert len(result) <= 200

    def test_newlines_replaced(self) -> None:
        text = "a\nb\nc"
        result = _source_context(text, 0, 5, window=0)
        assert "\n" not in result


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


class TestExtractClaims:
    def test_single_turn(self) -> None:
        records = [
            {"qa_id": "q1", "source_content_id": "s1", "answer": "Temp is 125°F and pressure 70 psig."}
        ]
        temp, pressure = extract_claims(records)
        assert len(temp) == 1
        assert len(pressure) == 1
        assert temp[0]["value"] == 125.0
        assert pressure[0]["value"] == 70.0

    def test_multi_turn(self) -> None:
        records = [
            {
                "qa_id": "q2",
                "source_content_id": "s2",
                "type": "multi_turn",
                "conversation": [
                    {"role": "user", "content": "What temp?"},
                    {"role": "assistant", "content": "It is 200°F."},
                    {"role": "user", "content": "Pressure?"},
                    {"role": "assistant", "content": "About 300 psig."},
                ],
            }
        ]
        temp, pressure = extract_claims(records)
        assert len(temp) == 1
        assert len(pressure) == 1

    def test_multi_turn_only_assistant_turns(self) -> None:
        records = [
            {
                "qa_id": "q3",
                "source_content_id": "s3",
                "type": "multi_turn",
                "conversation": [
                    {"role": "user", "content": "temp is 50°F?"},
                    {"role": "assistant", "content": "No, it is 100°F."},
                ],
            }
        ]
        temp, _ = extract_claims(records)
        # Only the assistant turn should be extracted (100°F not 50°F).
        assert len(temp) == 1
        assert temp[0]["value"] == 100.0

    def test_empty_records(self) -> None:
        temp, pressure = extract_claims([])
        assert temp == []
        assert pressure == []

    def test_no_measurements(self) -> None:
        records = [{"qa_id": "q4", "source_content_id": "s4", "answer": "No numbers here."}]
        temp, pressure = extract_claims(records)
        assert temp == []
        assert pressure == []

    def test_claim_fields(self) -> None:
        records = [{"qa_id": "q5", "source_content_id": "s5", "answer": "125°F"}]
        temp, _ = extract_claims(records)
        claim = temp[0]
        assert "qa_id" in claim
        assert "source_content_id" in claim
        assert "value" in claim
        assert "unit" in claim
        assert "claim_text" in claim
        assert "context" in claim
        assert "turn_index" in claim

    def test_negative_temperature_claim(self) -> None:
        records = [{"qa_id": "q6", "source_content_id": "s6", "answer": "Outside it was -10°C."}]
        temp, _ = extract_claims(records)
        assert len(temp) == 1
        assert temp[0]["value"] == -10.0

    def test_multiple_records(self) -> None:
        records = [
            {"qa_id": "q7", "source_content_id": "s7", "answer": "125°F"},
            {"qa_id": "q8", "source_content_id": "s8", "answer": "200°F and 300°F"},
        ]
        temp, _ = extract_claims(records)
        assert len(temp) == 3

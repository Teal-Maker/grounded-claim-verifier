"""Tests for grounded_claim_verifier.cli."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from grounded_claim_verifier.cli import sample_claims


# ---------------------------------------------------------------------------
# sample_claims
# ---------------------------------------------------------------------------


def _make_claim(value: float, unit: str = "°F", src_id: str = "s1") -> dict:
    return {
        "qa_id": "q1",
        "source_content_id": src_id,
        "value": value,
        "unit": unit,
        "claim_text": f"{value}{unit}",
        "context": "",
        "turn_index": 0,
    }


class TestSampleClaims:
    def test_samples_requested_counts(self) -> None:
        temps = [_make_claim(float(i), "°F") for i in range(50)]
        pressures = [_make_claim(float(i), "PSIG") for i in range(50)]
        sampled = sample_claims(temps, pressures, n_temp=10, n_pressure=10, seed=1)
        assert sum(1 for c in sampled if c["claim_type"] == "temperature") == 10
        assert sum(1 for c in sampled if c["claim_type"] == "pressure") == 10

    def test_falls_back_to_available(self) -> None:
        temps = [_make_claim(1.0)]
        pressures = [_make_claim(2.0, "PSIG")]
        sampled = sample_claims(temps, pressures, n_temp=25, n_pressure=25, seed=42)
        assert len(sampled) == 2

    def test_claim_type_set(self) -> None:
        temps = [_make_claim(1.0)]
        pressures = [_make_claim(2.0, "PSIG")]
        sampled = sample_claims(temps, pressures, n_temp=1, n_pressure=1, seed=42)
        types = {c["claim_type"] for c in sampled}
        assert types == {"temperature", "pressure"}

    def test_reproducible_with_same_seed(self) -> None:
        temps = [_make_claim(float(i)) for i in range(100)]
        pressures = [_make_claim(float(i), "PSIG") for i in range(100)]
        s1 = sample_claims(temps[:], pressures[:], seed=99)
        s2 = sample_claims(temps[:], pressures[:], seed=99)
        assert [c["value"] for c in s1] == [c["value"] for c in s2]

    def test_different_seeds_differ(self) -> None:
        temps = [_make_claim(float(i)) for i in range(100)]
        pressures = [_make_claim(float(i), "PSIG") for i in range(100)]
        s1 = sample_claims(temps, pressures, seed=1)
        s2 = sample_claims(temps, pressures, seed=2)
        assert [c["value"] for c in s1] != [c["value"] for c in s2]

    def test_empty_lists(self) -> None:
        sampled = sample_claims([], [], n_temp=10, n_pressure=10)
        assert sampled == []


# ---------------------------------------------------------------------------
# CLI integration (via subprocess to avoid argparse exit() conflicts)
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_run_with_jsonl_provider(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        fixture_dir = Path(__file__).parent / "fixtures"
        output = tmp_path / "results.json"

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grounded_claim_verifier.cli",
                "--input",
                str(fixture_dir / "qa_records.jsonl"),
                "--sources",
                str(fixture_dir / "sources.jsonl"),
                "--output",
                str(output),
                "--n-temp",
                "5",
                "--n-pressure",
                "5",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()

        data = json.loads(output.read_text())
        assert "hallucination_rate" in data
        assert "claims" in data
        assert isinstance(data["claims"], list)

    def test_run_no_provider_exits_nonzero(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        fixture_dir = Path(__file__).parent / "fixtures"
        output = tmp_path / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grounded_claim_verifier.cli",
                "--input",
                str(fixture_dir / "qa_records.jsonl"),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "DATABASE_URL": ""},
        )
        assert result.returncode != 0

    def test_mutually_exclusive_sources_and_db(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        fixture_dir = Path(__file__).parent / "fixtures"
        output = tmp_path / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grounded_claim_verifier.cli",
                "--input",
                str(fixture_dir / "qa_records.jsonl"),
                "--sources",
                str(fixture_dir / "sources.jsonl"),
                "--db-url",
                "postgresql://localhost/test",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

"""Tests for grounded_claim_verifier.providers."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from grounded_claim_verifier.providers.base import SourceProvider
from grounded_claim_verifier.providers.jsonl import JSONLProvider


# ---------------------------------------------------------------------------
# SourceProvider protocol
# ---------------------------------------------------------------------------


class TestSourceProviderProtocol:
    def test_conforming_class_is_instance(self) -> None:
        class GoodProvider:
            def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
                return {}

        assert isinstance(GoodProvider(), SourceProvider)

    def test_non_conforming_class_is_not_instance(self) -> None:
        class BadProvider:
            def get_texts(self, ids: list[str]) -> dict[str, str]:
                return {}

        assert not isinstance(BadProvider(), SourceProvider)


# ---------------------------------------------------------------------------
# JSONLProvider
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class TestJSONLProvider:
    def test_fetch_existing_ids(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [
            {"id": "a", "text": "Some text A"},
            {"id": "b", "text": "Some text B"},
        ])
        provider = JSONLProvider(src)
        result = provider.fetch_texts(["a", "b"])
        assert result == {"a": "Some text A", "b": "Some text B"}

    def test_missing_ids_absent_from_result(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"id": "x", "text": "text"}])
        provider = JSONLProvider(src)
        result = provider.fetch_texts(["x", "missing"])
        assert "missing" not in result
        assert "x" in result

    def test_empty_request(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"id": "a", "text": "text"}])
        provider = JSONLProvider(src)
        assert provider.fetch_texts([]) == {}

    def test_numeric_id_as_string(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"id": 42, "text": "numeric id"}])
        provider = JSONLProvider(src)
        result = provider.fetch_texts(["42"])
        assert result == {"42": "numeric id"}

    def test_custom_field_names(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"doc_id": "z", "body": "body text"}])
        provider = JSONLProvider(src, id_field="doc_id", text_field="body")
        result = provider.fetch_texts(["z"])
        assert result == {"z": "body text"}

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        src.write_text('\n{"id":"a","text":"hello"}\n\n', encoding="utf-8")
        provider = JSONLProvider(src)
        result = provider.fetch_texts(["a"])
        assert result == {"a": "hello"}

    def test_cache_used_on_second_call(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"id": "a", "text": "text"}])
        provider = JSONLProvider(src)
        provider.fetch_texts(["a"])  # populates cache
        # Overwrite the file — cached version should still be returned
        src.write_text('{"id":"a","text":"CHANGED"}\n', encoding="utf-8")
        result = provider.fetch_texts(["a"])
        assert result["a"] == "text"

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        src.write_text("not valid json\n", encoding="utf-8")
        provider = JSONLProvider(src)
        with pytest.raises(ValueError, match="Invalid JSON"):
            provider.fetch_texts(["anything"])

    def test_records_without_text_field_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [
            {"id": "a"},          # no text field
            {"id": "b", "text": "valid"},
        ])
        provider = JSONLProvider(src)
        result = provider.fetch_texts(["a", "b"])
        assert "a" not in result
        assert result["b"] == "valid"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        src = tmp_path / "sources.jsonl"
        _write_jsonl(src, [{"id": "a", "text": "text"}])
        provider = JSONLProvider(str(src))
        result = provider.fetch_texts(["a"])
        assert "a" in result

    def test_fixture_file(self) -> None:
        fixture = (
            Path(__file__).parent / "fixtures" / "sources.jsonl"
        )
        provider = JSONLProvider(fixture)
        result = provider.fetch_texts(["src-001", "src-002"])
        assert "src-001" in result
        assert "src-002" in result
        assert "125°F" in result["src-001"]


# ---------------------------------------------------------------------------
# DatabaseProvider import guard
# ---------------------------------------------------------------------------


class TestDatabaseProviderImportGuard:
    def test_raises_import_error_without_psycopg2(self, monkeypatch) -> None:
        import sys
        # Temporarily make psycopg2 unimportable
        monkeypatch.setitem(sys.modules, "psycopg2", None)

        # Re-import to trigger the guard
        import importlib
        import grounded_claim_verifier.providers.database as db_mod
        importlib.reload(db_mod)

        with pytest.raises(ImportError, match="psycopg2"):
            db_mod.DatabaseProvider("postgresql://localhost/test")

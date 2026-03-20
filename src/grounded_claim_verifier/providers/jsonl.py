"""JSONL-backed source text provider."""

from __future__ import annotations

import json
from pathlib import Path


class JSONLProvider:
    """Read source texts from a JSONL file.

    Each line of the file must be a JSON object with at minimum an ``id``
    field and a ``text`` field::

        {"id": "abc123", "text": "The suction pressure should be ..."}

    The entire file is loaded into memory on first access and then cached
    so that repeated ``fetch_texts`` calls are cheap.

    Parameters
    ----------
    path:
        Path to the JSONL source file.
    id_field:
        Name of the field used as the document identifier (default: ``"id"``).
    text_field:
        Name of the field containing the source text (default: ``"text"``).
    """

    def __init__(
        self,
        path: Path | str,
        id_field: str = "id",
        text_field: str = "text",
    ) -> None:
        self._path = Path(path)
        self._id_field = id_field
        self._text_field = text_field
        self._cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, str]:
        """Load and cache all records from the JSONL file."""
        if self._cache is not None:
            return self._cache

        cache: dict[str, str] = {}
        with open(self._path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno} of {self._path}: {exc}"
                    ) from exc
                doc_id = obj.get(self._id_field)
                text = obj.get(self._text_field)
                if doc_id is not None and text is not None:
                    cache[str(doc_id)] = str(text)

        self._cache = cache
        return self._cache

    # ------------------------------------------------------------------
    # SourceProvider protocol
    # ------------------------------------------------------------------

    def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
        """Return source texts for the requested IDs.

        Parameters
        ----------
        source_ids:
            List of IDs to look up.

        Returns
        -------
        dict[str, str]
            ``{id: text}`` for every ID present in the file.
        """
        store = self._load()
        return {sid: store[sid] for sid in source_ids if sid in store}

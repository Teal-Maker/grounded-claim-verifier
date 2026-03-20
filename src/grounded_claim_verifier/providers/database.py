"""PostgreSQL-backed source text provider (optional dependency)."""

from __future__ import annotations

import re

_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _validate_identifier(name: str, label: str) -> None:
    """Raise ValueError if *name* is not a safe SQL identifier.

    Accepts letters, digits, underscores, and dots (for schema-qualified names
    such as ``schema.table``).  Rejects anything else to prevent SQL injection
    via f-string identifier interpolation.
    """
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Unsafe SQL identifier for {label}: {name!r}")


class DatabaseProvider:
    """Fetch source texts from a PostgreSQL table via psycopg2.

    This provider requires the ``database`` optional dependency::

        pip install grounded-claim-verifier[database]

    Parameters
    ----------
    connection_string:
        A libpq connection string or DSN, e.g.
        ``"postgresql://user:pass@host:5432/dbname"``.
    table_name:
        Name of the table containing source documents (default: ``"sources"``).
    id_column:
        Column that holds the document ID (default: ``"id"``).
    text_column:
        Column that holds the source text (default: ``"text"``).
    batch_size:
        Number of IDs fetched per query (default: 100).

    Raises
    ------
    ImportError
        If ``psycopg2`` is not installed.
    ValueError
        If any of ``table_name``, ``id_column``, or ``text_column`` contains
        characters that are unsafe to interpolate into a SQL identifier position.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "sources",
        id_column: str = "id",
        text_column: str = "text",
        batch_size: int = 100,
    ) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for DatabaseProvider. "
                "Install it with: pip install grounded-claim-verifier[database]"
            ) from exc

        _validate_identifier(table_name, "table_name")
        _validate_identifier(id_column, "id_column")
        _validate_identifier(text_column, "text_column")

        self._connection_string = connection_string
        self._table_name = table_name
        self._id_column = id_column
        self._text_column = text_column
        self._batch_size = batch_size
        self._psycopg2 = psycopg2

    # ------------------------------------------------------------------
    # SourceProvider protocol
    # ------------------------------------------------------------------

    def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
        """Fetch source texts from the database for the given IDs.

        IDs are converted to the appropriate type by the database driver
        (integer columns will receive Python ``int`` values if the ID
        strings are numeric).  Non-numeric IDs are passed as strings.

        Parameters
        ----------
        source_ids:
            List of IDs to look up.

        Returns
        -------
        dict[str, str]
            ``{id: text}`` for every row found.  The key is always a string
            regardless of the underlying column type.
        """
        if not source_ids:
            return {}

        result: dict[str, str] = {}
        ids_list = list(source_ids)

        # Coerce to int when possible so integer PKs work without casting.
        def _coerce(v: str) -> int | str:
            try:
                return int(v)
            except ValueError:
                return v

        coerced = [_coerce(i) for i in ids_list]

        conn = self._psycopg2.connect(self._connection_string)
        try:
            with conn.cursor() as cur:
                for offset in range(0, len(coerced), self._batch_size):
                    chunk = coerced[offset : offset + self._batch_size]
                    placeholders = ",".join(["%s"] * len(chunk))
                    cur.execute(
                        f"SELECT {self._id_column}, {self._text_column}"
                        f" FROM {self._table_name}"
                        f" WHERE {self._id_column} IN ({placeholders})",
                        chunk,
                    )
                    for row in cur.fetchall():
                        if row[1] is not None:
                            result[str(row[0])] = row[1]
        finally:
            conn.close()

        return result

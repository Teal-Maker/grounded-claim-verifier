"""Base protocol for source text providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SourceProvider(Protocol):
    """Protocol for fetching source texts by ID.

    Implementors retrieve the original text associated with each source ID
    so that claims extracted from generated responses can be verified against
    the originating material.
    """

    def fetch_texts(self, source_ids: list[str]) -> dict[str, str]:
        """Fetch source texts by their IDs.

        Parameters
        ----------
        source_ids:
            List of opaque string IDs identifying the source documents.

        Returns
        -------
        dict[str, str]
            Mapping of ``{id: text}`` for every ID that was found.
            Missing IDs are simply absent from the result.
        """
        ...

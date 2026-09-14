from typing import Protocol

from pydantic import HttpUrl

from ..schemas.common import AOIBaseModel


class ResearchItem(AOIBaseModel):
    """Normalized search or source item retrieved by a research provider."""

    title: str
    url: HttpUrl
    content: str
    published_date: str | None = None
    raw_content: str | None = None
    score: float = 0.0


class ResearchProvider(Protocol):
    """Boundary between AOI research logic and external search/extraction providers."""

    def search(self, query: str, max_results: int = 5) -> list[ResearchItem]:
        """Retrieve relevant source material for a research query."""
        ...

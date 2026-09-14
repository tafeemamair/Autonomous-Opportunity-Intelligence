from typing import Protocol

from ..schemas.discovery import Candidate, DiscoveryPlan


class DiscoveryProvider(Protocol):
    """Boundary between AOI discovery logic and external search providers."""

    def discover(self, plan: DiscoveryPlan) -> list[Candidate]:
        ...

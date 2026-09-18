from unittest.mock import MagicMock, patch

import httpx
import pytest

from aoi.providers.exa import (
    ExaAuthenticationError,
    ExaConfigurationError,
    ExaDiscoveryProvider,
)
from aoi.schemas.discovery import DiscoveryPlan, DiscoveryStrategy
from aoi.agents.discovery import DiscoveryAgent
from aoi.schemas.objective import AOIInput, BusinessObjective


@pytest.fixture
def sample_plan() -> DiscoveryPlan:
    agent = DiscoveryAgent()
    aoi_input = AOIInput(
        objective=BusinessObjective(
            description="Find companies with recent AI automation needs.",
            target_markets=["United States"],
            target_opportunities=["AI automation"],
        )
    )
    full_plan = agent.plan(aoi_input)
    return DiscoveryPlan(
        strategies=[
            DiscoveryStrategy(
                name="hiring_signals",
                purpose=full_plan.strategies[0].purpose,
                query_templates=['"AI automation" "United States" hiring'],
            )
        ],
        created_at=full_plan.created_at,
    )


def test_exa_missing_key_raises(sample_plan):
    with patch.dict("os.environ", {}, clear=True):
        provider = ExaDiscoveryProvider(api_key=None)
        provider.api_key = None
        with pytest.raises(ExaConfigurationError, match="Exa API key is missing"):
            provider.discover(sample_plan)


def test_exa_maps_results_to_candidates(sample_plan):
    mock_client = MagicMock(spec=httpx.Client)
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "results": [
            {
                "title": "Anthropic - Careers and Research",
                "url": "https://www.anthropic.com/careers/research-engineer",
                "text": "Anthropic is hiring engineers to build frontier AI models and automation.",
            },
            {
                "title": "Staff AI Engineer - Acme Robotics - San Francisco, CA",
                "url": "https://www.indeed.com/viewjob?jk=12345",
                "text": "Acme Robotics is looking for a Staff AI Engineer to automate warehouse workflows.",
            },
        ]
    }
    mock_client.post.return_value = response

    provider = ExaDiscoveryProvider(api_key="exa-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    assert len(candidates) == 2
    anthropic = next(c for c in candidates if c.company_name == "Anthropic")
    acme = next(c for c in candidates if c.company_name == "Acme Robotics")
    assert str(anthropic.source_urls[0]) == "https://www.anthropic.com/careers/research-engineer"
    assert str(acme.source_urls[0]) == "https://www.indeed.com/viewjob?jk=12345"


def test_exa_retry_on_429(sample_plan):
    mock_client = MagicMock(spec=httpx.Client)

    rate_limited = MagicMock(spec=httpx.Response)
    rate_limited.status_code = 429
    rate_limited.text = "Rate limited"

    success = MagicMock(spec=httpx.Response)
    success.status_code = 200
    success.json.return_value = {
        "results": [
            {
                "title": "Retry Co - Systems",
                "url": "https://retry.example.com/about",
                "text": "Systems automation company.",
            }
        ]
    }
    mock_client.post.side_effect = [rate_limited, success]

    with patch("time.sleep", return_value=None):
        provider = ExaDiscoveryProvider(
            api_key="exa-mock-key",
            max_retries=2,
            backoff_factor=0.01,
            client=mock_client,
        )
        candidates = provider.discover(sample_plan)

    assert len(candidates) == 1
    assert candidates[0].company_name == "Retry Co"
    assert mock_client.post.call_count == 2


def test_exa_auth_error_401():
    mock_client = MagicMock(spec=httpx.Client)
    response = MagicMock(spec=httpx.Response)
    response.status_code = 401
    response.text = "Invalid API key"
    mock_client.post.return_value = response

    provider = ExaDiscoveryProvider(api_key="invalid-key", client=mock_client)
    with pytest.raises(ExaAuthenticationError, match="Exa authentication failed"):
        provider._search_with_retry("test query")

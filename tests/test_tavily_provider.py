from unittest.mock import MagicMock, patch

import httpx
import pytest

from aoi.agents.discovery import DiscoveryAgent
from aoi.cli import run_discovery
from aoi.providers.tavily import (
    TavilyAuthenticationError,
    TavilyConfigurationError,
    TavilyDiscoveryProvider,
)
from aoi.schemas.common import SourceType
from aoi.schemas.discovery import DiscoveryPlan, DiscoveryStrategy
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


def test_tavily_missing_key_raises(sample_plan):
    with patch.dict("os.environ", {}, clear=True):
        provider = TavilyDiscoveryProvider(api_key=None)
        provider.api_key = None
        with pytest.raises(TavilyConfigurationError, match="Tavily API key is missing"):
            provider.discover(sample_plan)


def test_tavily_direct_company_result_preserved(sample_plan):
    """Direct company websites continue to map correctly as candidate and website."""
    mock_response = {
        "results": [
            {
                "title": "Anthropic - Careers and Research",
                "url": "https://www.anthropic.com/careers/research-engineer",
                "content": "Anthropic is hiring engineers to build frontier AI models and automation.",
                "score": 0.95,
            },
            {
                "title": "Scale AI | Data Infrastructure",
                "url": "https://scale.com/jobs/ai-specialist",
                "content": "Scale is expanding AI data labeling and workflow automation.",
                "score": 0.88,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    assert len(candidates) == 2

    cand1 = next(c for c in candidates if "anthropic" in str(c.website).lower())
    assert cand1.company_name == "Anthropic"
    assert cand1.discovery_strategy == "hiring_signals"
    assert str(cand1.source_urls[0]) == "https://www.anthropic.com/careers/research-engineer"

    cand2 = next(c for c in candidates if "scale" in str(c.website).lower())
    assert cand2.company_name == "Scale AI"
    assert str(cand2.source_urls[0]) == "https://scale.com/jobs/ai-specialist"


def test_tavily_job_board_extracts_target_company(sample_plan):
    """Job-board result mentioning a target company -> target company becomes candidate, job board remains source."""
    mock_response = {
        "results": [
            {
                "title": "Staff AI Engineer - Acme Robotics - San Francisco, CA",
                "url": "https://www.indeed.com/viewjob?jk=12345",
                "content": "Acme Robotics is looking for a Staff AI Engineer to automate warehouse workflows.",
                "score": 0.92,
            },
            {
                "title": "Senior AI Developer at Bedrock Ocean Exploration",
                "url": "https://www.ziprecruiter.com/jobs/ai-developer-bedrock",
                "content": "Lead autonomous subsea AI workflows and infrastructure.",
                "score": 0.90,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    assert len(candidates) == 2

    # Verify Acme Robotics candidate
    acme_cand = next(c for c in candidates if c.company_name == "Acme Robotics")
    assert acme_cand.company_name == "Acme Robotics"
    # The source URL remains Indeed, but Indeed is NOT the company name
    assert str(acme_cand.source_urls[0]) == "https://www.indeed.com/viewjob?jk=12345"
    assert acme_cand.source_types[0] == SourceType.JOB_BOARD

    # Verify Bedrock Ocean Exploration candidate
    bedrock_cand = next(c for c in candidates if c.company_name == "Bedrock Ocean Exploration")
    assert bedrock_cand.company_name == "Bedrock Ocean Exploration"
    assert str(bedrock_cand.source_urls[0]) == "https://www.ziprecruiter.com/jobs/ai-developer-bedrock"
    assert bedrock_cand.source_types[0] == SourceType.JOB_BOARD


def test_tavily_news_article_extracts_target_company(sample_plan):
    """News/article result mentioning a company action -> mentioned company becomes candidate, publication remains source."""
    mock_response = {
        "results": [
            {
                "title": "Sierra Raises $350M to Build Enterprise AI Agents | TechCrunch",
                "url": "https://techcrunch.com/2026/05/10/sierra-ai-agents-enterprise",
                "content": "Sierra has closed $350M in funding to deploy conversational AI agents for enterprises.",
                "score": 0.91,
            },
            {
                "title": "BlueVoyant Launches First Microsoft AI Agent Security Service",
                "url": "https://www.reuters.com/technology/bluevoyant-ai-security-service",
                "content": "BlueVoyant announced a new managed service for enterprise AI agent protection.",
                "score": 0.89,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    assert len(candidates) == 2

    # Sierra candidate extracted, TechCrunch remains source
    sierra_cand = next(c for c in candidates if c.company_name == "Sierra")
    assert sierra_cand.company_name == "Sierra"
    assert str(sierra_cand.source_urls[0]) == "https://techcrunch.com/2026/05/10/sierra-ai-agents-enterprise"
    assert sierra_cand.source_types[0] == SourceType.NEWS

    # BlueVoyant candidate extracted, Reuters remains source
    bv_cand = next(c for c in candidates if c.company_name == "BlueVoyant")
    assert bv_cand.company_name == "BlueVoyant"
    assert str(bv_cand.source_urls[0]) == "https://www.reuters.com/technology/bluevoyant-ai-security-service"
    assert bv_cand.source_types[0] == SourceType.NEWS


def test_tavily_skips_government_and_info_pages(sample_plan):
    """Government, regulatory, academic, and encyclopedia pages without a target company are skipped."""
    mock_response = {
        "results": [
            {
                "title": "AI agent - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/AI_agent",
                "content": "An AI agent is an autonomous entity that directs its activity towards achieving goals.",
                "score": 0.85,
            },
            {
                "title": 'Announcing the "AI Agent Standards Initiative" for Interoperable and Secure AI | NIST',
                "url": "https://www.nist.gov/news-events/news/2026/02/announcing-ai-agent-standards-initiative",
                "content": "NIST announces the launch of the AI Agent Standards Initiative.",
                "score": 0.83,
            },
            {
                "title": "USAJOBS - Search",
                "url": "https://ai.usajobs.gov/search",
                "content": "Search official government job openings in the United States.",
                "score": 0.80,
            },
            {
                "title": "Request for Information Regarding Security Considerations for AI Agents",
                "url": "https://www.federalregister.gov/documents/2026/01/08/2026-00206/request-for-information",
                "content": "A Notice by the National Institute of Standards and Technology.",
                "score": 0.79,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    # All reference/government pages must be skipped
    assert len(candidates) == 0


def test_tavily_skips_listicle_and_directory_index(sample_plan):
    """Directory/listicle page where the publisher itself is not the target company -> skip unless single company reliably extracted."""
    mock_response = {
        "results": [
            {
                "title": "Top AI Automation Companies in the USA (2026) · BrainBox",
                "url": "https://www.brainboxautomations.com/blogs/top-ai-automation-companies-usa",
                "content": "LeewayHertz, Azumo, and more top companies evaluated.",
                "score": 0.82,
            },
            {
                "title": "433 Agentic AI Startups (2026) — Recently Funded | VCBacked",
                "url": "https://www.vcbacked.co/directory/industries/agentic-ai",
                "content": "Directory of 433 recently funded agentic AI startups in the US.",
                "score": 0.80,
            },
            {
                "title": "Top 156 AI Startups 2026 | Funded by Sequoia, YC, A16Z",
                "url": "https://topstartups.io/?industries=Artificial+Intelligence",
                "content": "Browse 156 AI startups with active hiring rounds.",
                "score": 0.78,
            },
            {
                "title": "$23-$39/hr Ai Agent Jobs (NOW HIRING) Sep 2026 | ZipRecruiter",
                "url": "https://www.ziprecruiter.com/Jobs/Ai-Agent",
                "content": "Browse 1000+ AI AGENT jobs ($23-$39/hr).",
                "score": 0.77,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    # All generic listicles and aggregator index pages must be skipped
    assert len(candidates) == 0


def test_tavily_domain_deduplication(sample_plan):
    """Existing domain deduplication continues to work and merges source URLs."""
    mock_response = {
        "results": [
            {
                "title": "Acme Corp - Job 1",
                "url": "https://acme.com/jobs/job-1",
                "content": "First hiring posting.",
                "score": 0.9,
            },
            {
                "title": "Acme Corp - Job 2",
                "url": "https://acme.com/jobs/job-2",
                "content": "Second hiring posting.",
                "score": 0.85,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = mock_response
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)
    candidates = provider.discover(sample_plan)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.company_name == "Acme Corp"
    assert len(candidate.source_urls) == 2


def test_tavily_retry_on_429(sample_plan):
    mock_client = MagicMock(spec=httpx.Client)

    mock_429 = MagicMock(spec=httpx.Response)
    mock_429.status_code = 429
    mock_429.text = "Rate limited"

    mock_200 = MagicMock(spec=httpx.Response)
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "results": [
            {
                "title": "Retry Co - Systems",
                "url": "https://retry.io/about",
                "content": "Systems automation company.",
                "score": 0.8,
            }
        ]
    }

    mock_client.post.side_effect = [mock_429, mock_200]

    with patch("time.sleep", return_value=None):
        provider = TavilyDiscoveryProvider(
            api_key="tvly-mock-key",
            max_retries=2,
            backoff_factor=0.01,
            client=mock_client,
        )
        candidates = provider.discover(sample_plan)

    assert len(candidates) == 1
    assert candidates[0].company_name == "Retry Co"
    assert mock_client.post.call_count == 2


def test_tavily_auth_error_401():
    mock_client = MagicMock(spec=httpx.Client)
    mock_401 = MagicMock(spec=httpx.Response)
    mock_401.status_code = 401
    mock_401.text = "Invalid API key"
    mock_client.post.return_value = mock_401

    provider = TavilyDiscoveryProvider(api_key="invalid-key", client=mock_client)
    with pytest.raises(TavilyAuthenticationError, match="Tavily authentication failed"):
        provider._search_with_retry("test query")


def test_cli_runner_missing_key(capsys):
    with patch("aoi.providers.tavily.get_settings") as mock_settings, patch.dict("os.environ", {}, clear=True):
        mock_settings.return_value.tavily_api_key = None
        code = run_discovery(
            objective="Find companies with recent AI automation needs.",
            api_key=None,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "Tavily API key is missing" in captured.err


def test_cli_runner_success(capsys):
    mock_candidates = [
        DiscoveryAgent().normalize_candidate(
            company_name="Mock Corp",
            discovery_strategy="hiring_signals",
            discovery_reason="Discovered via hiring query.",
            website="https://mock.example.com",
            source_urls=["https://mock.example.com/careers"],
        )
    ]

    with patch.object(TavilyDiscoveryProvider, "discover", return_value=mock_candidates):
        code = run_discovery(
            objective="Find companies with recent AI automation needs.",
            api_key="tvly-mock-key",
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "Discovered 1 unique candidates" in captured.out
        assert "Mock Corp" in captured.out

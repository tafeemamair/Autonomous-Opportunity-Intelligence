from datetime import UTC, datetime, timedelta

from pydantic import HttpUrl

from aoi.agents.research import ResearchAgent
from aoi.graph.workflow import aoi_graph, create_initial_state
from aoi.providers.research import ResearchItem, ResearchProvider
from aoi.schemas.common import RunStatus, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.objective import AOIInput, BusinessObjective
from aoi.schemas.research import ResearchResult, ResearchStatus


class DummyResearchProvider(ResearchProvider):
    """Deterministic mock provider for unit testing."""

    def __init__(self, responses: dict[str, list[ResearchItem]] | None = None):
        self.responses = responses or {}
        self.recorded_queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[ResearchItem]:
        self.recorded_queries.append(query)
        for key, items in self.responses.items():
            if key.lower() in query.lower():
                return items[:max_results]
        return []


class FailingResearchProvider(ResearchProvider):
    """Mock provider that always raises an error."""

    def search(self, query: str, max_results: int = 5) -> list[ResearchItem]:
        raise ConnectionError("Search service unavailable")


def test_research_agent_generates_valid_queries():
    candidate = Candidate(
        candidate_id="cand-1",
        company_name="Bedrock Ocean Exploration",
        website=HttpUrl("https://bedrockocean.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="Found hiring posting",
        discovered_at=datetime.now(UTC),
    )
    agent = ResearchAgent(provider=DummyResearchProvider())
    queries = agent.generate_research_queries(candidate)

    assert len(queries) == 7
    assert any('"Bedrock Ocean Exploration" official website' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" "AI"' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" "AI agents"' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" automation' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" hiring jobs' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" engineering' in q for q in queries)
    assert any('"Bedrock Ocean Exploration" announcement launch funding' in q for q in queries)


def test_research_result_schema_validates():
    result = ResearchResult(
        candidate_id="cand-1",
        company_name="Bedrock Ocean Exploration",
        website=HttpUrl("https://bedrockocean.com"),
        location="Richmond, CA",
        industry="Autonomous Marine Robotics",
        people=[],
        technology_signals=["Building autonomous seafloor survey subsea vehicles with AI agents."],
        business_signals=["Hiring Staff AI Platform Engineers."],
        problem_signals=["Requires scalable autonomous data processing pipelines."],
        evidence=[],
        warnings=[],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    assert result.candidate_id == "cand-1"
    assert result.company_name == "Bedrock Ocean Exploration"
    assert result.research_status == ResearchStatus.COMPLETED
    assert result.location == "Richmond, CA"


def test_evidence_extraction_creates_structured_evidence():
    candidate = Candidate(
        candidate_id="cand-1",
        company_name="Bedrock Ocean Exploration",
        discovery_strategy="job_search",
        discovery_reason="Job listing for AI Engineer",
        source_urls=[HttpUrl("https://jobright.ai/jobs/123")],
        source_types=[SourceType.JOB_BOARD],
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "Bedrock Ocean Exploration": [
                ResearchItem(
                    title="Bedrock Ocean Careers",
                    url=HttpUrl("https://bedrockocean.com/careers"),
                    content="Bedrock Ocean Exploration is hiring a Staff AI Platform Engineer for AI agent data pipelines.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=2)
    result = agent.research(candidate)

    assert len(result.evidence) >= 2
    # First evidence item is the discovery origin
    disc_evidence = result.evidence[0]
    assert disc_evidence.source.source_type == SourceType.JOB_BOARD
    assert "jobright" in str(disc_evidence.source.publisher).lower()

    # Check that the careers research produced a structured evidence claim
    assert any("Staff AI Platform Engineer" in ev.claim for ev in result.evidence)
    career_evidence = next(ev for ev in result.evidence if "Staff AI Platform Engineer" in ev.claim)
    assert career_evidence.source.source_type == SourceType.JOB_BOARD
    assert career_evidence.verification_status == VerificationStatus.UNVERIFIED
    assert career_evidence.recency_score > 50


def test_source_type_and_reliability_assignment():
    agent = ResearchAgent(provider=DummyResearchProvider())

    # Tier 1
    assert agent._infer_source_type("https://acme.com/press-releases/launch", "acme.com") == SourceType.OFFICIAL_ANNOUNCEMENT
    assert agent._compute_reliability(SourceType.COMPANY_WEBSITE) == 95
    assert agent._compute_reliability(SourceType.OFFICIAL_JOB) == 90

    # Tier 2
    assert agent._infer_source_type("https://techcrunch.com/2026/01/ai", "techcrunch.com") == SourceType.NEWS
    assert agent._compute_reliability(SourceType.NEWS) == 80
    assert agent._infer_source_type("https://venturebeat.com/ai/article", "venturebeat.com") == SourceType.INDUSTRY_PUBLICATION
    assert agent._compute_reliability(SourceType.INDUSTRY_PUBLICATION) == 75

    # Tier 3
    assert agent._infer_source_type("https://linkedin.com/jobs/view/123", "linkedin.com") == SourceType.LINKEDIN
    assert agent._compute_reliability(SourceType.LINKEDIN) == 65
    assert agent._infer_source_type("https://boards.greenhouse.io/acme/jobs/456", "greenhouse.io") == SourceType.JOB_BOARD
    assert agent._compute_reliability(SourceType.JOB_BOARD) == 60

    # Tier 4 & 5
    assert agent._compute_reliability(SourceType.DIRECTORY) == 45
    assert agent._compute_reliability(SourceType.SOCIAL) == 25


def test_published_date_and_recency_handling():
    agent = ResearchAgent(provider=DummyResearchProvider(), recency_days=90)
    now = datetime.now(UTC)

    # Within 90 days
    recent_date = now - timedelta(days=20)
    assert agent._compute_recency_score(recent_date) == 95

    # Within 180 days
    mid_date = now - timedelta(days=120)
    assert agent._compute_recency_score(mid_date) == 75

    # Within 365 days
    older_date = now - timedelta(days=250)
    assert agent._compute_recency_score(older_date) == 50

    # Older than 1 year
    very_old_date = now - timedelta(days=500)
    assert agent._compute_recency_score(very_old_date) == 30

    # No date provided
    assert agent._compute_recency_score(None) == 50


def test_unsupported_claims_are_not_created():
    candidate = Candidate(
        candidate_id="cand-unsupported",
        company_name="Apex Robotics",
        discovery_strategy="news",
        discovery_reason="Found in tech news",
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "Apex Robotics": [
                ResearchItem(
                    title="Apex Robotics Update",
                    url=HttpUrl("https://apexrobotics.com/news"),
                    content="Apex Robotics is expanding operations in Chicago.",
                    published_date="2026-08-10",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=1)
    result = agent.research(candidate)

    # Should NOT contain speculative or inferential claim statements
    for ev in result.evidence:
        assert "definitely a client" not in ev.claim
        assert "needs our services" not in ev.claim
        assert "must purchase" not in ev.claim

    # People should be empty because no leadership was mentioned
    assert len(result.people) == 0


def test_missing_company_website_does_not_crash_research():
    candidate = Candidate(
        candidate_id="cand-no-website",
        company_name="DeepSea Intelligence",
        website=None,
        discovery_strategy="search",
        discovery_reason="Discovered through search query",
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "DeepSea Intelligence": [
                ResearchItem(
                    title="DeepSea Intelligence Home",
                    url=HttpUrl("https://deepseaintelligence.com/"),
                    content="DeepSea Intelligence builds autonomous marine sensory equipment.",
                    published_date=None,
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=1)
    result = agent.research(candidate)

    assert result.website == HttpUrl("https://deepseaintelligence.com/")
    assert result.research_status != ResearchStatus.FAILED


def test_search_failure_produces_warning_and_status_not_crash():
    candidate = Candidate(
        candidate_id="cand-fail",
        company_name="ErrorProne Corp",
        website=HttpUrl("https://errorprone.com"),
        discovery_strategy="search",
        discovery_reason="Found via search",
        discovered_at=datetime.now(UTC),
    )
    failing_provider = FailingResearchProvider()
    agent = ResearchAgent(provider=failing_provider, max_queries=2)

    result = agent.research(candidate)

    assert result.research_status == ResearchStatus.FAILED
    assert len(result.warnings) > 0
    assert any("Search service unavailable" in w for w in result.warnings)
    assert result.company_name == "ErrorProne Corp"


def test_multiple_evidence_items_can_be_attached():
    candidate = Candidate(
        candidate_id="cand-multi",
        company_name="Oceanic AI",
        website=HttpUrl("https://oceanic.ai"),
        discovery_strategy="hiring_signal",
        discovery_reason="Job posting found",
        source_urls=[HttpUrl("https://jobsite.com/oceanic")],
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "Oceanic AI": [
                ResearchItem(
                    title="Oceanic AI Careers",
                    url=HttpUrl("https://oceanic.ai/careers"),
                    content="Oceanic AI is hiring a Lead AI Engineer to automate repetitive tasks.",
                    published_date="2026-08-01",
                ),
                ResearchItem(
                    title="About Oceanic AI",
                    url=HttpUrl("https://oceanic.ai/about"),
                    content="Founded by Sarah Connor, Oceanic AI is based in Boston.",
                    published_date="2026-07-01",
                ),
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=1)
    result = agent.research(candidate)

    assert len(result.evidence) >= 3
    # Check that Sarah Connor is extracted without hallucination
    assert any(p.name == "Sarah Connor" and p.role == "Founder" for p in result.people)
    assert result.location == "Boston"


def test_graph_executes_research_node_with_candidates():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI needs."))
    state = create_initial_state(input_data)

    cand = Candidate(
        candidate_id="cand-graph-1",
        company_name="RoboNav",
        website=HttpUrl("https://robonav.ai"),
        discovery_strategy="hiring",
        discovery_reason="Found open AI role",
        source_urls=[HttpUrl("https://jobs.robonav.ai/1")],
        discovered_at=datetime.now(UTC),
    )
    state = state.model_copy(update={"candidates": [cand]})

    result = aoi_graph.invoke(state)

    assert result["status"] in (
        RunStatus.RESEARCHING,
        RunStatus.QUALIFYING,
        RunStatus.VERIFYING,
        RunStatus.SCORING,
        RunStatus.COMPLETED,
    )
    assert len(result["research_results"]) == 1
    assert result["research_results"][0].company_name == "RoboNav"


def test_bedrock_ocean_exploration_acceptance_scenario():
    """Verify Section 14 Acceptance Criteria:

    Candidate: Bedrock Ocean Exploration
    Discovery source: Jobright job listing
    Research independently establishes identity, AI/engineering signals, leadership,
    and factual evidence without inferential claims.
    """
    candidate = Candidate(
        candidate_id="cand-bedrock",
        company_name="Bedrock Ocean Exploration",
        website=None,  # Not initially known from job board
        discovery_strategy="job_search",
        discovery_reason="Jobright listing: Staff AI Platform Engineer",
        source_urls=[HttpUrl("https://jobright.ai/jobs/bedrock-ocean-exploration")],
        source_types=[SourceType.JOB_BOARD],
        discovered_at=datetime.now(UTC),
    )

    provider = DummyResearchProvider(
        responses={
            "official website": [
                ResearchItem(
                    title="Bedrock Ocean Exploration - Autonomous Seafloor Mapping",
                    url=HttpUrl("https://bedrockocean.com/"),
                    content="Bedrock Ocean Exploration is headquartered in Richmond, CA. Founded by Anthony DiMare, building autonomous underwater vehicles for ocean mapping.",
                    published_date="2026-07-15",
                )
            ],
            "AI": [
                ResearchItem(
                    title="Bedrock Careers - Staff AI Platform Engineer",
                    url=HttpUrl("https://jobs.ashbyhq.com/bedrockocean/123"),
                    content="Bedrock Ocean Exploration has an open role for a Staff AI Platform Engineer to build autonomous agent data processing systems and internal tooling.",
                    published_date="2026-08-20",
                )
            ],
        }
    )

    agent = ResearchAgent(provider=provider, max_queries=2)
    res = agent.research(candidate)

    # 1. Company identity established
    assert res.company_name == "Bedrock Ocean Exploration"
    assert res.website == HttpUrl("https://bedrockocean.com/")
    assert res.location == "Richmond, CA"

    # 2. People established without hallucination
    assert any(p.name == "Anthony DiMare" and p.role == "Founder" for p in res.people)

    # 3. Signals established
    assert any("autonomous agent" in s.lower() or "ai" in s.lower() for s in res.technology_signals)
    assert any("staff ai platform engineer" in s.lower() for s in res.business_signals)

    # 4. Source vs Claim separation:
    # Source is Jobright or Ashbyhq, claim is factual about the role/tech
    claims = [ev.claim for ev in res.evidence]
    assert any("Bedrock Ocean Exploration discovered via job_search" in c for c in claims)
    assert any("hiring" in c.lower() or "role" in c.lower() for c in claims)

    # 5. Must NOT contain speculative qualification statements
    for c in claims:
        assert "definitely a client" not in c
        assert "needs our services" not in c

    assert res.research_status == ResearchStatus.COMPLETED


def test_domain_source_type_classification_regression():
    agent = ResearchAgent(provider=DummyResearchProvider())

    # Social platforms
    assert agent._infer_source_type("https://www.youtube.com/watch?v=123", "youtube.com") == SourceType.SOCIAL
    assert agent._infer_source_type("https://www.instagram.com/company", "instagram.com") == SourceType.SOCIAL
    assert agent._infer_source_type("https://twitter.com/company", "twitter.com") == SourceType.SOCIAL
    assert agent._infer_source_type("https://x.com/company", "x.com") == SourceType.SOCIAL

    # Directories
    assert agent._infer_source_type("https://clutch.co/profile/acme", "clutch.co") == SourceType.DIRECTORY
    assert agent._infer_source_type("https://leadiq.com/c/acme/123", "leadiq.com") == SourceType.DIRECTORY
    assert agent._infer_source_type("https://www.g2.com/products/acme", "g2.com") == SourceType.DIRECTORY

    # Official Announcements / Press Wires
    assert agent._infer_source_type("https://www.prnewswire.com/news-releases/article-123.html", "prnewswire.com") == SourceType.OFFICIAL_ANNOUNCEMENT
    assert agent._infer_source_type("https://www.businesswire.com/news/home/2026/article", "businesswire.com") == SourceType.OFFICIAL_ANNOUNCEMENT

    # Unknown domains do NOT default to COMPANY_WEBSITE
    assert agent._infer_source_type("https://dust.tt/blog/article", "dust.tt", company_name="Glean") == SourceType.OTHER
    assert agent._infer_source_type("https://random-thirdparty.org/info", "random-thirdparty.org") == SourceType.OTHER

    # Known candidate company website
    assert agent._infer_source_type("https://www.glean.com/about", "glean.com", company_name="Glean") == SourceType.COMPANY_WEBSITE
    assert agent._infer_source_type("https://glean.com/pricing", "glean.com", official_domain="glean.com") == SourceType.COMPANY_WEBSITE


def test_decision_maker_extraction_glean_regression():
    agent = ResearchAgent(provider=DummyResearchProvider())
    content = "At Glean, Toby Roberts\n\nSenior VP of Engineering is responsible for platform scalability."
    people = agent._extract_people(content, company_name="Glean", source_url=HttpUrl("https://glean.com/team"))

    assert len(people) == 1
    person = people[0]
    assert person.name == "Toby Roberts"
    assert person.role == "Senior VP of Engineering"


def test_location_extraction_bluevoyant_regression():
    candidate = Candidate(
        candidate_id="cand-bluevoyant-loc",
        company_name="BlueVoyant",
        website=HttpUrl("https://www.bluevoyant.com"),
        discovery_strategy="search",
        discovery_reason="Security automation",
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "BlueVoyant": [
                ResearchItem(
                    title="About BlueVoyant",
                    url=HttpUrl("https://www.bluevoyant.com/about"),
                    content="BlueVoyant is headquartered in New York City, the company has over 600 employees across offices in College Park, Washington, D.C.",
                    published_date=None,
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=1)
    res = agent.research(candidate)

    assert res.location == "New York City"


def test_stock_image_caption_noise_filtering_cgi_regression():
    candidate = Candidate(
        candidate_id="cand-cgi-stock",
        company_name="CGI United States",
        website=HttpUrl("https://www.cgi.com/us/en-us"),
        discovery_strategy="search",
        discovery_reason="AI services",
        discovered_at=datetime.now(UTC),
    )
    provider = DummyResearchProvider(
        responses={
            "CGI United States": [
                ResearchItem(
                    title="CGI Careers",
                    url=HttpUrl("https://www.cgi.com/careers"),
                    content="Young Female Artificial Intelligence Engineer Working on Computer in a Technological Office",
                    published_date=None,
                ),
                ResearchItem(
                    title="CGI AI Initiatives",
                    url=HttpUrl("https://www.cgi.com/ai-agents"),
                    content="CGI United States helps federal agencies deploy autonomous AI agents to automate complex workflows.",
                    published_date=None,
                ),
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=2)
    res = agent.research(candidate)

    # Obvious stock image caption must NOT be included in business signals, problem signals, or evidence
    for sig in res.business_signals:
        assert "Young Female" not in sig
        assert "Working on Computer" not in sig

    for sig in res.problem_signals:
        assert "Young Female" not in sig
        assert "Working on Computer" not in sig

    for ev in res.evidence:
        assert "Young Female" not in ev.claim
        assert "Working on Computer" not in ev.claim

    # Legitimate agentic workflow signals should be preserved
    assert any("AI agents" in s for s in res.technology_signals)


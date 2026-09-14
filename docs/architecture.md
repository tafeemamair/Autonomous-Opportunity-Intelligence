# AOI V1 Architecture

```text
Business Objective
        |
        v
  Orchestrator
        |
        v
 Discovery Agent
        |
        v
 Candidate set
        |
        +--> Research Agent (next milestone)
        +--> Qualification Agent (next milestone)
        +--> Verification Agent (next milestone)
        |
        v
 Ranked Report
        |
        v
 Human decision
```

## Schema-first rule

Domain objects are Pydantic models. Database persistence uses SQLAlchemy models. LangGraph state is typed and carries the workflow snapshot.

External integrations are adapters, not domain logic. Search providers should return AOI-native candidate/source objects rather than leaking provider-specific payloads into the core domain.

## Discovery Agent in this milestone

The Discovery Agent does not call an LLM or search API yet. It validates the objective and converts it into deterministic discovery strategies and query templates. The provider boundary will be added before a live search integration.

## Safety boundary

No outbound communication, proposal submission, purchasing, or other consequential external action belongs in the V1 research graph.

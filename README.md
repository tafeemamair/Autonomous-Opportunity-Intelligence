# Autonomous Opportunity Intelligence (AOI)

AOI is a human-supervised autonomous opportunity research system.

## V1 objective

Given a structured business objective, AOI will eventually discover candidate companies, research them, collect evidence, identify potential opportunities, score and verify them, and produce a ranked decision-ready report.

V1 does **not** send outreach, submit applications, spend money, or take consequential external actions.

## Current milestone

This repository starts with strict contracts before external integrations:

- Pydantic domain schemas
- LangGraph state
- SQLAlchemy database models
- deterministic discovery planning
- fixture-free local graph execution
- configuration and environment handling
- tests

External LLM/search integrations are intentionally not connected yet.

## Setup

Python 3.11+ is required. `uv` is recommended.

```bash
uv venv
uv pip install -e '.[dev]'
cp .env.example .env
pytest
```

## Structure

```text
aoi/
├── src/aoi/
│   ├── agents/discovery.py
│   ├── config.py
│   ├── db/{base.py,models.py}
│   ├── graph/{state.py,workflow.py}
│   └── schemas/{common.py,objective.py,discovery.py,opportunity.py}
├── tests/
├── docs/architecture.md
├── .env.example
├── .gitignore
├── langgraph.json
└── pyproject.toml
```

## Design rule

> AOI can investigate autonomously. AOI cannot act externally without human approval.

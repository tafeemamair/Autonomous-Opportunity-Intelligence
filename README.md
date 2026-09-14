# Autonomous Opportunity Intelligence (AOI)

Autonomous Opportunity Intelligence (AOI) is an agentic system designed to discover, research, qualify, and score high-value business opportunities.

For architecture and specifications, see [docs/architecture.md](docs/architecture.md).

## Development Setup

Requires Python 3.11+.

```bash
# Install package with development dependencies
pip install -e ".[dev]"
```

## Configuring Tavily API Key

AOI uses Tavily as its first real discovery search provider. Configure your API key via environment variable or in a `.env` file in the project root:

```bash
# In shell:
export TAVILY_API_KEY="tvly-your-api-key-here"

# On Windows PowerShell:
$env:TAVILY_API_KEY="tvly-your-api-key-here"
```

Or in `.env`:
```env
TAVILY_API_KEY=tvly-your-api-key-here
```

> **Warning**: Running discovery against the live Tavily provider performs real HTTP requests to the Tavily search API and **incurs provider usage and API costs**. Unit tests run entirely offline with mocked responses and do not use your API key or quota.

## Running Discovery Locally

You can execute a discovery cycle using the CLI entry point:

```bash
# Run with default objective
python -m aoi.cli discover

# Or specify a custom business objective and market
python -m aoi.cli discover \
  --objective "Find B2B companies adopting AI customer support workflows" \
  --market "United States" \
  --opportunity "AI customer support" \
  --max-results 3
```

You can also pass `--api-key tvly-...` directly on the command line if preferred.

## Running Verification Checks

### 1. Run Tests (Offline / Mocked)
```bash
pytest
```

### 2. Run Linter
```bash
ruff check .
```

### 3. Compile Validation
```bash
python -m compileall src tests
```

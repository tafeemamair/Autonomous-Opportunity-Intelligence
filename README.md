# Autonomous Opportunity Intelligence (AOI)

Autonomous Opportunity Intelligence (AOI) is an agentic system designed to discover, research, qualify, and score high-value business opportunities.

For architecture and specifications, see [docs/architecture.md](docs/architecture.md).

## Development Setup

Requires Python 3.11+.

```bash
# Install package with development dependencies
pip install -e ".[dev]"
```

## Running Verification Checks

### 1. Run Tests
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

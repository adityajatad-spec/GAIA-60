# gaia-agent

A CLI research agent that answers complex, multi-step questions using tool use — web search (Tavily), web browsing, code execution, and file parsing.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) (recommended) or venv
- API keys for **Anthropic** (Claude) and **Tavily** (search)

## Setup

```bash
cd gaia-agent
cp .env.example .env        # → add your real API keys

# With Poetry
poetry install && poetry shell

# Or with venv
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Usage

```bash
python -m agent "What is the capital of France?"

# With attached files (file_read tool not yet implemented)
python -m agent -f report.pdf "Summarise the key findings"

# Override model or step budget
python -m agent -m claude-sonnet-4-20250514 -s 20 "Your question"
```

The agent prints each step live: model reasoning, tool calls, and results. When finished it writes a full JSON trajectory to `logs/<timestamp>.json`.

## Configuration

All config is via `.env` in the project root:

| Variable           | Required | Description              |
|--------------------|----------|--------------------------|
| `ANTHROPIC_API_KEY`| Yes      | Anthropic API key        |
| `TAVILY_API_KEY`   | Yes      | Tavily search API key    |
| `ANTHROPIC_MODEL`  | No       | Model name (default: claude-sonnet-4-20250514) |

## Project Structure

```
agent/         # Main agent package
  __main__.py  # CLI entry point, live printer, JSON logger
  config.py    # dotenv config loading
  core.py      # ResearchAgent — tool-use loop, trajectory tracking
tools/         # Tool implementations
  web.py       # web_search (Tavily), web_browse (trafilatura)
eval/          # (empty — no evaluation harness yet)
logs/          # Runtime trajectory JSON files (gitignored)
tests/         # Tests
  test_core.py # 4 unit tests (mocked Anthropic)
  test_web.py  # 10 integration tests (5 browse, 5 search with TAVILY_API_KEY)
```

## Implementation Status

| Phase | Component            | Status           |
|-------|----------------------|------------------|
| 0     | Project skeleton     | ✅ Complete      |
| 1     | Core agent loop      | ✅ Complete      |
| 2a    | `web_search` tool    | ✅ Complete      |
| 2b    | `web_browse` tool    | ✅ Complete      |
| 2c    | CLI + live logging   | ✅ Complete      |
| 3     | `code_exec` tool     | ❌ Stub ("NOT IMPLEMENTED") |
| 4     | `file_read` tool     | ❌ Stub ("NOT IMPLEMENTED") |
| 5     | Evaluation harness   | ❌ Not started   |

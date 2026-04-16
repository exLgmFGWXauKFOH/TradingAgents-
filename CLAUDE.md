# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install the package and all dependencies
pip install .

# Run the interactive CLI
tradingagents
python -m cli.main     # alternative from source

# Run a single analysis programmatically
python main.py

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_model_validation.py

# Run with Docker
cp .env.example .env   # add API keys first
docker compose run --rm tradingagents
```

## Architecture

TradingAgents is a LangGraph-based multi-agent pipeline that processes a ticker + date through a fixed sequence of teams. The main entry point is `TradingAgentsGraph` in `tradingagents/graph/trading_graph.py`.

### Graph Execution Flow

```
Analysts (parallel reports, sequential in graph)
  → Market / Social / News / Fundamentals Analyst
  → Bull Researcher ↔ Bear Researcher (debate, N rounds)
  → Research Manager (judge)
  → Trader
  → Aggressive ↔ Conservative ↔ Neutral Analyst (risk debate, N rounds)
  → Portfolio Manager → END
```

The graph is compiled in `tradingagents/graph/setup.py` (`GraphSetup.setup_graph`). Analysts are optional — the `selected_analysts` list controls which are included. Between each analyst and the next, messages are cleared to keep context manageable (`create_msg_delete` in `agent_utils.py`).

### State Schema

`AgentState` (in `tradingagents/agents/utils/agent_states.py`) extends LangGraph's `MessagesState` and carries flat report strings (`market_report`, `sentiment_report`, etc.) plus two nested debate states:
- `InvestDebateState` — bull/bear/judge histories and round counter
- `RiskDebateState` — aggressive/conservative/neutral/judge histories and round counter

`propagation.py` creates the initial blank state; `signal_processing.py` parses the final `final_trade_decision` string into a normalized signal.

### Data Vendor Routing

All data tool calls go through `tradingagents/dataflows/interface.py`. `route_to_vendor(method, ...)` looks up the configured vendor per category (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) with tool-level overrides, then dispatches to the matching yfinance or Alpha Vantage implementation. Alpha Vantage rate-limit errors trigger automatic fallback to the next vendor.

The agent-facing tools (e.g., `get_stock_data`, `get_news`) are defined as LangChain tools in `tradingagents/agents/utils/` (one file per category) and imported into the `ToolNode` dicts in `TradingAgentsGraph._create_tool_nodes`.

### LLM Clients

`tradingagents/llm_clients/factory.py` maps provider names to client classes (all extend `BaseLLMClient`). OpenAI, xAI, DeepSeek, Qwen, GLM, Ollama, and OpenRouter all share `OpenAIClient` via the OpenAI-compatible chat completions API. Anthropic and Google have their own clients. Provider-specific thinking/effort options are forwarded as kwargs.

The model catalog (`llm_clients/model_catalog.py`) is the single source of truth for CLI model lists and model validation; `validators.py` warns (but does not error) for unknown models against providers with fixed catalogs.

### Memory

`FinancialSituationMemory` (in `tradingagents/agents/utils/memory.py`) uses BM25 (rank-bm25) for retrieval. There are five per-session instances: bull, bear, trader, invest_judge, portfolio_manager. `TradingAgentsGraph.reflect_and_remember(returns)` populates them after a trade. Memory is in-memory only — not persisted between runs.

### Configuration

`tradingagents/default_config.py` is the single config dict. Key fields:
- `llm_provider` — openai, anthropic, google, xai, deepseek, qwen, glm, ollama, openrouter, azure
- `deep_think_llm` / `quick_think_llm` — model IDs for complex vs. fast nodes
- `max_debate_rounds` / `max_risk_discuss_rounds` — controls iteration counts
- `data_vendors` — category-level vendor selection
- `tool_vendors` — per-tool overrides (take precedence)
- `output_language` — affects analyst reports; internal debate always stays in English
- Results and cache default to `~/.tradingagents/logs` and `~/.tradingagents/cache`; override with `TRADINGAGENTS_RESULTS_DIR` / `TRADINGAGENTS_CACHE_DIR`

Enterprise providers (Azure OpenAI, AWS Bedrock) are configured in `.env.enterprise` (loaded automatically by `cli/main.py`).

### CLI

`cli/main.py` drives an interactive questionnaire (using `questionary` + `typer`), then streams the LangGraph execution through a `rich.Live` layout showing agent status, tool calls, and live report sections. `StatsCallbackHandler` (`cli/stats_handler.py`) is a LangChain callback handler that tracks LLM call counts and token usage for the footer display.

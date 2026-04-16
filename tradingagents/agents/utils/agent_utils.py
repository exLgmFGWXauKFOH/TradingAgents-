"""Agent utilities: pure helpers (eager) + data tool re-exports (lazy).

The three pure utilities are imported eagerly from agent_utils_core so that
lightweight code paths (e.g. tests) don't pay the cost of loading the full
data stack (yfinance, pandas, etc.).  Data tools are imported on first access.
"""
from tradingagents.agents.utils.agent_utils_core import (
    get_language_instruction,
    build_instrument_context,
    create_msg_delete,
)

__all__ = [
    "get_language_instruction",
    "build_instrument_context",
    "create_msg_delete",
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_insider_transactions",
    "get_global_news",
]

_LAZY_TOOLS = {
    "get_stock_data": "tradingagents.agents.utils.core_stock_tools",
    "get_indicators": "tradingagents.agents.utils.technical_indicators_tools",
    "get_fundamentals": "tradingagents.agents.utils.fundamental_data_tools",
    "get_balance_sheet": "tradingagents.agents.utils.fundamental_data_tools",
    "get_cashflow": "tradingagents.agents.utils.fundamental_data_tools",
    "get_income_statement": "tradingagents.agents.utils.fundamental_data_tools",
    "get_news": "tradingagents.agents.utils.news_data_tools",
    "get_insider_transactions": "tradingagents.agents.utils.news_data_tools",
    "get_global_news": "tradingagents.agents.utils.news_data_tools",
}


def __getattr__(name: str):
    if name in _LAZY_TOOLS:
        import importlib
        module = importlib.import_module(_LAZY_TOOLS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

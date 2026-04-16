from .utils.agent_utils_core import create_msg_delete
from .utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from .utils.memory import FinancialSituationMemory

__all__ = [
    "FinancialSituationMemory",
    "AgentState",
    "create_msg_delete",
    "InvestDebateState",
    "RiskDebateState",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_research_manager",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_neutral_debator",
    "create_news_analyst",
    "create_aggressive_debator",
    "create_portfolio_manager",
    "create_conservative_debator",
    "create_social_media_analyst",
    "create_trader",
]

_LAZY = {
    "create_fundamentals_analyst": "tradingagents.agents.analysts.fundamentals_analyst",
    "create_market_analyst": "tradingagents.agents.analysts.market_analyst",
    "create_news_analyst": "tradingagents.agents.analysts.news_analyst",
    "create_social_media_analyst": "tradingagents.agents.analysts.social_media_analyst",
    "create_bear_researcher": "tradingagents.agents.researchers.bear_researcher",
    "create_bull_researcher": "tradingagents.agents.researchers.bull_researcher",
    "create_aggressive_debator": "tradingagents.agents.risk_mgmt.aggressive_debator",
    "create_conservative_debator": "tradingagents.agents.risk_mgmt.conservative_debator",
    "create_neutral_debator": "tradingagents.agents.risk_mgmt.neutral_debator",
    "create_research_manager": "tradingagents.agents.managers.research_manager",
    "create_portfolio_manager": "tradingagents.agents.managers.portfolio_manager",
    "create_trader": "tradingagents.agents.trader.trader",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

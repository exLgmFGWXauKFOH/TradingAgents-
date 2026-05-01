"""
TradingAgents — Stock Analysis App
Analyze any ticker with the full multi-agent pipeline: Analysts → Research → Trader → Risk → Portfolio Manager.

Run with:
    streamlit run trading_analysis_app.py
"""

import os
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

st.set_page_config(
    page_title="TradingAgents — Stock Analysis",
    page_icon="📊",
    layout="wide",
)

load_dotenv()

# Pull Streamlit Cloud secrets into env vars
if hasattr(st, "secrets"):
    for _k in ("ANTHROPIC_API_KEY",):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]

# ── Analyst options ───────────────────────────────────────────────────────────

ANALYST_OPTIONS = {
    "fundamentals": "Fundamentals Analyst — financial statements, valuation ratios",
    "news": "News Analyst — recent headlines and events",
    "social": "Social Sentiment — Reddit and social media signals",
    "market": "Market Analyst — technical indicators, price action",
}

SIGNAL_STYLE = {
    "BUY":         ("#d4edda", "#28a745", "#155724"),
    "OVERWEIGHT":  ("#d4edda", "#28a745", "#155724"),
    "HOLD":        ("#fff3cd", "#ffc107", "#856404"),
    "UNDERWEIGHT": ("#f8d7da", "#dc3545", "#721c24"),
    "SELL":        ("#f8d7da", "#dc3545", "#721c24"),
}

# ── Analysis runner ───────────────────────────────────────────────────────────

def run_analysis(ticker, trade_date, selected_analysts, debate_rounds, write_progress=None):
    if write_progress is None:
        write_progress = lambda _: None

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-6"
    config["quick_think_llm"] = "claude-haiku-4-5"
    config["max_debate_rounds"] = debate_rounds
    config["max_risk_discuss_rounds"] = debate_rounds
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    ta = TradingAgentsGraph(selected_analysts=selected_analysts, config=config)
    write_progress("Agents initialized. Starting research pipeline...")

    init_state = ta.propagator.create_initial_state(ticker, trade_date)
    graph_args = ta.propagator.get_graph_args()

    milestone_map = {
        "market_report": "Market analysis complete",
        "fundamentals_report": "Fundamentals analysis complete",
        "news_report": "News analysis complete",
        "sentiment_report": "Sentiment analysis complete",
        "trader_investment_plan": "Trader plan complete",
        "final_trade_decision": "Final decision reached",
    }
    milestones_done = set()
    last_chunk = None

    for chunk in ta.graph.stream(init_state, **graph_args):
        for key, label in milestone_map.items():
            if chunk.get(key) and key not in milestones_done:
                write_progress(f"✅ {label}")
                milestones_done.add(key)

        invest_state = chunk.get("investment_debate_state") or {}
        if invest_state.get("judge_decision") and "research_judge" not in milestones_done:
            write_progress("✅ Research team decision complete")
            milestones_done.add("research_judge")

        risk_state = chunk.get("risk_debate_state") or {}
        if risk_state.get("judge_decision") and "risk_judge" not in milestones_done:
            write_progress("✅ Risk management decision complete")
            milestones_done.add("risk_judge")

        last_chunk = chunk

    if last_chunk is None:
        raise RuntimeError("Analysis produced no output.")

    signal = ta.process_signal(last_chunk.get("final_trade_decision", ""))
    return last_chunk, signal


# ── UI helpers ────────────────────────────────────────────────────────────────

def signal_banner(signal_text: str):
    action = signal_text.strip().upper()
    bg, border, fg = SIGNAL_STYLE.get(action, ("#e2e3e5", "#6c757d", "#383d41"))
    st.markdown(
        f"""<div style="background:{bg};border-left:6px solid {border};
        padding:16px 20px;border-radius:6px;margin-bottom:16px;">
        <span style="color:{fg};font-size:1.4rem;font-weight:700;">{action}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def section_expander(title, content, default_open=False):
    if content and content.strip():
        with st.expander(title, expanded=default_open):
            st.markdown(content)


def display_results(state, signal):
    st.subheader("Final Decision")
    raw_signal = signal if isinstance(signal, str) else str(signal)
    signal_banner(raw_signal)

    st.subheader("Analysis Reports")
    tabs_config = [
        ("Fundamentals", state.get("fundamentals_report")),
        ("News", state.get("news_report")),
        ("Social Sentiment", state.get("sentiment_report")),
        ("Market / Technical", state.get("market_report")),
        ("Research Team", state.get("investment_plan") or _extract_investment_plan(state)),
        ("Trader Plan", state.get("trader_investment_plan")),
        ("Risk Management", _extract_risk_report(state)),
        ("Portfolio Manager", _extract_portfolio_decision(state)),
    ]
    tabs_config = [(label, content) for label, content in tabs_config if content and content.strip()]

    if not tabs_config:
        st.warning("No report sections found in analysis output.")
        return

    tab_labels = [label for label, _ in tabs_config]
    tabs = st.tabs(tab_labels)
    for tab, (_, content) in zip(tabs, tabs_config):
        with tab:
            st.markdown(content)


def _extract_investment_plan(state):
    debate = state.get("investment_debate_state") or {}
    parts = []
    if debate.get("bull_history"):
        parts.append(f"### Bull Researcher\n{debate['bull_history']}")
    if debate.get("bear_history"):
        parts.append(f"### Bear Researcher\n{debate['bear_history']}")
    if debate.get("judge_decision"):
        parts.append(f"### Research Manager Decision\n{debate['judge_decision']}")
    return "\n\n".join(parts) if parts else None


def _extract_risk_report(state):
    risk = state.get("risk_debate_state") or {}
    parts = []
    if risk.get("aggressive_history"):
        parts.append(f"### Aggressive Analyst\n{risk['aggressive_history']}")
    if risk.get("conservative_history"):
        parts.append(f"### Conservative Analyst\n{risk['conservative_history']}")
    if risk.get("neutral_history"):
        parts.append(f"### Neutral Analyst\n{risk['neutral_history']}")
    return "\n\n".join(parts) if parts else None


def _extract_portfolio_decision(state):
    risk = state.get("risk_debate_state") or {}
    decision = risk.get("judge_decision") or state.get("final_trade_decision")
    return decision


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("TradingAgents — Stock Analysis")
    st.caption(
        "Multi-agent AI framework: Analysts → Bull/Bear Research → Trader → "
        "Risk Management → Portfolio Manager"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        # API key
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            st.success("Anthropic API key loaded")
        else:
            entered_key = st.text_input("Anthropic API Key", type="password")
            if entered_key:
                os.environ["ANTHROPIC_API_KEY"] = entered_key
                st.success("API key set")

        st.divider()

        # Ticker
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="NYSE/NASDAQ: AAPL, NVDA, SPY. Toronto: CNQ.TO. Tokyo: 7203.T",
        ).upper().strip()

        # Date
        analysis_date = st.date_input(
            "Analysis Date",
            value=date.today() - timedelta(days=1),
            max_value=date.today(),
            help="Use yesterday or earlier — today's market data may be incomplete.",
        ).strftime("%Y-%m-%d")

        st.divider()

        # Analysts
        st.subheader("Analyst Team")
        selected_analysts = []
        defaults = {"fundamentals": True, "news": True, "social": True, "market": False}
        for key, label in ANALYST_OPTIONS.items():
            short_label, description = label.split(" — ", 1)
            if st.checkbox(short_label, value=defaults[key], help=description, key=f"a_{key}"):
                selected_analysts.append(key)

        st.divider()

        # Research depth
        st.subheader("Research Depth")
        debate_rounds = st.slider(
            "Debate rounds",
            min_value=1, max_value=3, value=1,
            help="Each round adds one Bull vs Bear exchange. More rounds = deeper analysis, longer runtime.",
        )
        runtime_estimate = {1: "3–6 min", 2: "6–10 min", 3: "10–16 min"}
        st.caption(f"Estimated runtime: {runtime_estimate[debate_rounds]}")

        st.divider()

        run_btn = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not selected_analysts or not ticker,
        )

    # ── Main area ─────────────────────────────────────────────────────────────

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.result_ticker = None
        st.session_state.result_date = None

    if run_btn:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Please provide your Anthropic API key in the sidebar.")
            st.stop()

        status_box = st.empty()
        progress_lines = []

        def write_progress(msg):
            progress_lines.append(msg)
            status_box.info("\n\n".join(progress_lines))

        write_progress(f"Starting analysis: **{ticker}** on {analysis_date}")
        write_progress(f"Analysts: {', '.join(selected_analysts)} | Debate rounds: {debate_rounds}")

        try:
            state, signal = run_analysis(
                ticker, analysis_date, selected_analysts, debate_rounds, write_progress
            )
            st.session_state.result = (state, signal)
            st.session_state.result_ticker = ticker
            st.session_state.result_date = analysis_date
            status_box.success("Analysis complete!")
        except Exception as exc:
            status_box.error(f"Analysis failed: {exc}")
            with st.expander("Error details"):
                st.exception(exc)
            st.stop()

    if st.session_state.result:
        state, signal = st.session_state.result
        t = st.session_state.result_ticker
        d = st.session_state.result_date
        st.caption(f"Results for **{t}** — {d}")
        display_results(state, signal)
    else:
        # Welcome / how-it-works screen
        st.markdown("### How it works")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**1. Analyst Team** gathers data in parallel:
- Fundamentals: income statement, balance sheet, P/E, margins
- News: recent headlines from Yahoo Finance
- Social: Reddit sentiment and trending discussions
- Market: RSI, MACD, Bollinger Bands, moving averages

**2. Research Team** debates the data:
- Bull Researcher argues for buying
- Bear Researcher argues for selling
- Research Manager synthesizes a position
""")
        with col2:
            st.markdown("""
**3. Trader** converts the research into a concrete trading plan

**4. Risk Management** stress-tests the plan:
- Aggressive Analyst pushes for larger positions
- Conservative Analyst flags risks
- Neutral Analyst balances both views

**5. Portfolio Manager** issues the final BUY / HOLD / SELL decision

---
Configure the ticker and analysts in the sidebar, then click **Run Analysis**.
""")


if __name__ == "__main__":
    main()

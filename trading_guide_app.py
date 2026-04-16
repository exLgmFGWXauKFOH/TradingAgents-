"""
Trading Guide — Beginner-friendly stock analysis + portfolio manager.
Powered by TradingAgents multi-agent AI framework with Anthropic Claude.

Run with:
    streamlit run trading_guide_app.py
"""

import json
import os
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

# Pull secrets into env vars (Streamlit Cloud stores them in st.secrets)
if hasattr(st, "secrets"):
    for _k in ("ANTHROPIC_API_KEY",):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]

# ── Constants ─────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = os.path.join(
    os.path.expanduser("~"), ".tradingagents", "portfolio.json"
)

SIGNAL_STYLE = {
    "BUY":         ("#d4edda", "#28a745", "#155724"),
    "OVERWEIGHT":  ("#d4edda", "#28a745", "#155724"),
    "HOLD":        ("#fff3cd", "#ffc107", "#856404"),
    "UNDERWEIGHT": ("#f8d7da", "#dc3545", "#721c24"),
    "SELL":        ("#f8d7da", "#dc3545", "#721c24"),
}
SIGNAL_ICON = {
    "BUY": "🟢", "OVERWEIGHT": "🟢",
    "HOLD": "🟡",
    "UNDERWEIGHT": "🔴", "SELL": "🔴",
}
SIGNAL_EXPLAIN = {
    "BUY": (
        "The AI agents believe this stock is currently **undervalued or has strong "
        "upward momentum**. This could be a good time to consider buying — but always "
        "do your own research and invest only what you can afford to lose."
    ),
    "OVERWEIGHT": (
        "The AI agents recommend holding **more of this stock** than its market "
        "weighting. Think of it as a soft 'Buy' — the stock looks attractive relative "
        "to the broader market."
    ),
    "HOLD": (
        "The AI agents see **roughly balanced** reasons to buy and sell. If you already "
        "own it, consider keeping it. If you don't, there may be better opportunities."
    ),
    "UNDERWEIGHT": (
        "The AI agents suggest holding **less of this stock** than its market weighting. "
        "Think of it as a soft 'Sell' — some caution is warranted."
    ),
    "SELL": (
        "The AI agents see **significant risks or overvaluation**. Based on the research, "
        "this may not be a good time to hold this stock."
    ),
}

# ── Portfolio helpers ─────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"positions": []}


def save_portfolio(data: dict) -> None:
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def upsert_position(ticker: str, shares: float, buy_price: float, buy_date: str) -> None:
    """Add a new position or update an existing one's shares / price / date."""
    data = load_portfolio()
    for pos in data["positions"]:
        if pos["ticker"] == ticker:
            pos["shares"] = shares
            pos["buy_price"] = buy_price
            pos["buy_date"] = buy_date
            save_portfolio(data)
            return
    data["positions"].append({
        "ticker": ticker,
        "shares": shares,
        "buy_price": buy_price,
        "buy_date": buy_date,
        "last_signal": None,
        "last_analysis_date": None,
        "last_updated": None,
        "fundamentals_report": "",
        "news_report": "",
        "sentiment_report": "",
        "invest_judge_decision": "",
        "final_trade_decision": "",
        "risk_judge_decision": "",
    })
    save_portfolio(data)


def save_analysis_to_position(ticker: str, results: dict) -> None:
    """Write the latest analysis results into a position's portfolio record."""
    data = load_portfolio()
    for pos in data["positions"]:
        if pos["ticker"] == ticker:
            pos["last_signal"] = results["signal"]
            pos["last_analysis_date"] = results["trade_date"]
            pos["last_updated"] = datetime.now().isoformat(timespec="seconds")
            pos["fundamentals_report"] = results.get("fundamentals_report", "")
            pos["news_report"] = results.get("news_report", "")
            pos["sentiment_report"] = results.get("sentiment_report", "")
            pos["invest_judge_decision"] = results.get("invest_judge_decision", "")
            pos["final_trade_decision"] = results.get("final_trade_decision", "")
            pos["risk_judge_decision"] = results.get("risk_judge_decision", "")
            break
    save_portfolio(data)


def remove_position(ticker: str) -> None:
    data = load_portfolio()
    data["positions"] = [p for p in data["positions"] if p["ticker"] != ticker]
    save_portfolio(data)


# ── Analysis helper ───────────────────────────────────────────────────────────

def run_ta_analysis(ticker: str, trade_date: str, write_progress=None) -> dict:
    """
    Run TradingAgents analysis and return a results dict.
    write_progress: optional callable(str) used to emit live progress messages.
    """
    if write_progress is None:
        write_progress = lambda _: None

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-6"
    config["quick_think_llm"] = "claude-sonnet-4-6"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    ta = TradingAgentsGraph(
        selected_analysts=["fundamentals", "news", "social"],
        config=config,
    )
    write_progress("Agents initialized. Starting research...")

    init_state = ta.propagator.create_initial_state(ticker, trade_date)
    graph_args = ta.propagator.get_graph_args()

    milestones = {
        "fundamentals_report": False,
        "news_report": False,
        "sentiment_report": False,
        "invest_judge": False,
        "final_trade_decision": False,
    }
    last_chunk = None

    for chunk in ta.graph.stream(init_state, **graph_args):
        if chunk.get("fundamentals_report") and not milestones["fundamentals_report"]:
            write_progress("✅ Fundamentals analysis complete")
            milestones["fundamentals_report"] = True
        if chunk.get("news_report") and not milestones["news_report"]:
            write_progress("✅ News analysis complete")
            milestones["news_report"] = True
        if chunk.get("sentiment_report") and not milestones["sentiment_report"]:
            write_progress("✅ Sentiment analysis complete")
            milestones["sentiment_report"] = True
        invest_state = chunk.get("investment_debate_state") or {}
        if invest_state.get("judge_decision") and not milestones["invest_judge"]:
            write_progress("✅ Bull vs. Bear debate complete")
            milestones["invest_judge"] = True
        if chunk.get("final_trade_decision") and not milestones["final_trade_decision"]:
            write_progress("✅ Final recommendation ready")
            milestones["final_trade_decision"] = True
        last_chunk = chunk

    signal = ta.process_signal(last_chunk["final_trade_decision"])
    risk_state = last_chunk.get("risk_debate_state") or {}
    invest_state = last_chunk.get("investment_debate_state") or {}

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "signal": signal.strip().upper(),
        "fundamentals_report": last_chunk.get("fundamentals_report", ""),
        "news_report": last_chunk.get("news_report", ""),
        "sentiment_report": last_chunk.get("sentiment_report", ""),
        "invest_judge_decision": invest_state.get("judge_decision", ""),
        "final_trade_decision": last_chunk.get("final_trade_decision", ""),
        "risk_judge_decision": risk_state.get("judge_decision", ""),
    }


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def signal_badge_html(signal: str) -> str:
    """Return an inline HTML badge for a signal string."""
    if not signal:
        return '<span style="color:#6c757d; font-style:italic;">Not analyzed</span>'
    bg, border, text = SIGNAL_STYLE.get(signal, ("#e2e3e5", "#6c757d", "#383d41"))
    icon = SIGNAL_ICON.get(signal, "⚪")
    return (
        f'<span style="background:{bg}; border:1px solid {border}; color:{text}; '
        f'padding:2px 10px; border-radius:6px; font-weight:700; font-size:0.95em;">'
        f"{icon} {signal}</span>"
    )


def render_analysis_results(r: dict) -> None:
    """Render the full analysis result dict (banner + tabs). Used in both tabs."""
    signal = r.get("signal", "")
    bg, border, text = SIGNAL_STYLE.get(signal, ("#e2e3e5", "#6c757d", "#383d41"))
    icon = SIGNAL_ICON.get(signal, "⚪")

    st.markdown(
        f"""
        <div style="background-color:{bg}; border:2px solid {border};
                    border-radius:12px; padding:20px; text-align:center; margin-bottom:16px;">
            <div style="font-size:2.4rem; font-weight:800; color:{text}; margin:0;">
                {icon}&nbsp;{signal}
            </div>
            <div style="font-size:1rem; color:{text}; margin-top:4px;">
                AI recommendation for <strong>{r['ticker']}</strong>
                as of <strong>{r['trade_date']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What does this mean for a beginner?", expanded=True):
        st.markdown(SIGNAL_EXPLAIN.get(signal, "See the report tabs below."))

    tab_fund, tab_news, tab_rec = st.tabs([
        "📊 Company Fundamentals",
        "📰 News & Sentiment",
        "🤖 Full AI Recommendation",
    ])

    with tab_fund:
        st.markdown(
            "_Fundamentals describe a company's financial health — revenue, profits, "
            "debt, and valuation. Strong fundamentals usually indicate a well-run company._"
        )
        st.divider()
        st.markdown(r["fundamentals_report"] or "_No fundamentals report generated._")

    with tab_news:
        st.markdown(
            "_News shows what's happening around the company. Sentiment reflects the "
            "general mood of investors and the public._"
        )
        st.divider()
        if r.get("news_report"):
            st.markdown("**News Report**")
            st.markdown(r["news_report"])
        if r.get("sentiment_report"):
            st.divider()
            st.markdown("**Sentiment Report**")
            st.markdown(r["sentiment_report"])
        if not r.get("news_report") and not r.get("sentiment_report"):
            st.markdown("_No news or sentiment data generated._")

    with tab_rec:
        st.markdown(
            "_The final decision after fundamentals, news, a bull vs. bear debate, "
            "and a risk management review._"
        )
        st.divider()
        st.markdown(r.get("final_trade_decision") or "_No recommendation generated._")
        if r.get("invest_judge_decision"):
            with st.expander("Bull vs. Bear Debate — Judge's Decision"):
                st.markdown(r["invest_judge_decision"])
        if r.get("risk_judge_decision"):
            with st.expander("Risk Assessment — Risk Team's Verdict"):
                st.markdown(r["risk_judge_decision"])


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Trading Guide",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AI Trading Guide")
    st.markdown(
        """
        **Analyze any stock** with AI agents, then **track your positions**
        in the Portfolio ledger.

        **What the AI examines:**
        - 📊 Fundamentals — earnings, revenue, valuation
        - 📰 News & Sentiment — headlines, public mood
        - 🤖 Bull vs. Bear Debate — AI agents argue both sides
        - ⚖️ Risk Assessment — final risk-adjusted verdict

        **Result:** BUY / HOLD / SELL with full reasoning.

        ---
        **Common tickers**

        | Company | Ticker |
        |---------|--------|
        | Apple | AAPL |
        | Tesla | TSLA |
        | Microsoft | MSFT |
        | Nvidia | NVDA |
        | Amazon | AMZN |
        | Google | GOOGL |
        | Meta | META |

        ---
        ⚠️ *Educational purposes only. Not financial advice.*
        """
    )

# ── Top-level tabs ────────────────────────────────────────────────────────────

tab_analyze, tab_portfolio = st.tabs(["🔍 Analyze Stock", "💼 My Portfolio"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE STOCK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analyze:
    st.header("Analyze a Stock")
    st.markdown(
        "Enter a ticker and date. AI agents will research the company and produce "
        "a structured trading guide."
    )

    # ── API key status ────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        if st.button("🔑 Test API Key", key="btn_test_key"):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Say OK"}],
                )
                st.success(f"API key works! Response: {msg.content[0].text}")
            except Exception as e:
                st.error(f"API key test failed: {e}")
    else:
        st.warning("ANTHROPIC_API_KEY not set — add it to Streamlit secrets.")

    col_ticker, col_date, col_btn = st.columns([2, 2, 1])
    with col_ticker:
        ticker_input = st.text_input(
            "Stock Ticker",
            value="AAPL",
            placeholder="e.g. AAPL",
            help="The ticker symbol (e.g. AAPL for Apple, TSLA for Tesla).",
            key="analyze_ticker",
        ).upper().strip()
    with col_date:
        yesterday = date.today() - timedelta(days=1)
        date_input = st.date_input(
            "Analysis Date",
            value=yesterday,
            max_value=yesterday,
            help="Use yesterday or earlier — real-time data has a 1-day delay.",
            key="analyze_date",
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("🔍 Analyze", type="primary",
                                    use_container_width=True, key="btn_analyze")

    if analyze_clicked:
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "**ANTHROPIC_API_KEY not found.**  \n"
                "Add it to a `.env` file: `ANTHROPIC_API_KEY=sk-ant-...`"
            )
        else:
            with st.status(
                f"Researching **{ticker_input}** for {date_input}...", expanded=True
            ) as status:
                try:
                    results = run_ta_analysis(
                        ticker_input, str(date_input), write_progress=st.write
                    )
                    st.session_state["guide_results"] = results
                    status.update(label="Research complete!", state="complete",
                                  expanded=False)
                except Exception as exc:
                    import traceback
                    status.update(label="Analysis failed", state="error")
                    st.error(f"**Error:** {exc}")
                    st.code(traceback.format_exc(), language="text")

    # ── Results ───────────────────────────────────────────────────────────────

    if "guide_results" in st.session_state:
        r = st.session_state["guide_results"]
        st.divider()
        render_analysis_results(r)

        # ── Add to Portfolio ──────────────────────────────────────────────────
        st.divider()
        with st.expander("➕ Add this position to My Portfolio"):
            st.markdown(
                f"Log your **{r['ticker']}** holding. The current analysis "
                f"({r['signal']}) will be saved with the position."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                add_shares = st.number_input(
                    "Shares held", min_value=0.0001, value=1.0, step=0.5,
                    key="add_shares"
                )
            with c2:
                add_price = st.number_input(
                    "Buy price per share ($)", min_value=0.01, value=100.0, step=1.0,
                    key="add_price"
                )
            with c3:
                add_buy_date = st.date_input(
                    "Date purchased", value=yesterday, max_value=date.today(),
                    key="add_buy_date"
                )
            if st.button("Save to Portfolio", key="btn_add_to_portfolio"):
                upsert_position(r["ticker"], add_shares, add_price, str(add_buy_date))
                save_analysis_to_position(r["ticker"], r)
                st.success(
                    f"**{r['ticker']}** added to your portfolio with the current "
                    f"analysis saved."
                )

        st.divider()
        st.caption(
            "⚠️ AI-generated for educational purposes only. Not financial advice. "
            "Consult a qualified financial advisor before investing."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("My Portfolio")
    st.markdown(
        "Track your holdings here. Click **Update** on any position to run a fresh "
        "AI analysis and refresh its signal."
    )

    portfolio = load_portfolio()
    positions = portfolio.get("positions", [])

    # ── Summary metrics ───────────────────────────────────────────────────────

    if positions:
        total = len(positions)
        signals = [p.get("last_signal") for p in positions]
        bullish = sum(1 for s in signals if s in ("BUY", "OVERWEIGHT"))
        bearish = sum(1 for s in signals if s in ("SELL", "UNDERWEIGHT"))
        neutral = sum(1 for s in signals if s == "HOLD")
        unanalyzed = sum(1 for s in signals if not s)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Positions", total)
        m2.metric("🟢 Bullish", bullish)
        m3.metric("🟡 Hold", neutral)
        m4.metric("🔴 Bearish", bearish)
        m5.metric("⚪ Unanalyzed", unanalyzed)
        st.divider()

    # ── Update All button ─────────────────────────────────────────────────────

    if positions:
        update_all_clicked = st.button(
            "🔄 Update All Positions", key="btn_update_all",
            help="Run a fresh AI analysis on every position using yesterday's date."
        )
        if update_all_clicked:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY not found. Add it to your .env file.")
            else:
                for pos in positions:
                    t = pos["ticker"]
                    with st.status(f"Updating **{t}**...", expanded=True) as s:
                        try:
                            results = run_ta_analysis(
                                t,
                                str(date.today() - timedelta(days=1)),
                                write_progress=st.write,
                            )
                            save_analysis_to_position(t, results)
                            s.update(label=f"{t} updated — {results['signal']}",
                                     state="complete", expanded=False)
                        except Exception as exc:
                            s.update(label=f"{t} failed", state="error")
                            st.error(str(exc))
                st.rerun()

    # ── Position cards ────────────────────────────────────────────────────────

    if not positions:
        st.info(
            "Your portfolio is empty. Analyze a stock in the **Analyze Stock** tab "
            "and use **Add to Portfolio**, or add a position manually below."
        )
    else:
        for pos in positions:
            t = pos["ticker"]
            signal = pos.get("last_signal") or ""
            last_updated = pos.get("last_updated", "")
            updated_label = (
                f"Last updated: {last_updated[:10]}" if last_updated else "Not yet analyzed"
            )

            with st.container(border=True):
                # Header row
                hcol_left, hcol_right = st.columns([3, 1])
                with hcol_left:
                    st.markdown(
                        f"### {t} &nbsp; {signal_badge_html(signal)}",
                        unsafe_allow_html=True,
                    )
                    st.caption(updated_label)
                with hcol_right:
                    # Confirm-before-remove pattern using session state
                    confirm_key = f"confirm_remove_{t}"
                    if st.session_state.get(confirm_key):
                        st.warning("Remove this position?")
                        cr1, cr2 = st.columns(2)
                        with cr1:
                            if st.button("Yes, remove", key=f"yes_remove_{t}",
                                         type="primary"):
                                remove_position(t)
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                        with cr2:
                            if st.button("Cancel", key=f"cancel_remove_{t}"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                    else:
                        if st.button("🗑 Remove", key=f"remove_{t}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

                # Details row
                d1, d2, d3 = st.columns(3)
                d1.markdown(f"**Shares:** {pos.get('shares', '—')}")
                d2.markdown(
                    f"**Buy price:** ${pos.get('buy_price', 0):,.2f}"
                    if pos.get("buy_price") else "**Buy price:** —"
                )
                d3.markdown(f"**Buy date:** {pos.get('buy_date', '—')}")

                # Update button row
                ucol, _ = st.columns([1, 3])
                with ucol:
                    if st.button(f"🔄 Update {t}", key=f"update_{t}",
                                 use_container_width=True):
                        if not os.environ.get("ANTHROPIC_API_KEY"):
                            st.error("ANTHROPIC_API_KEY not found.")
                        else:
                            with st.status(
                                f"Updating **{t}**...", expanded=True
                            ) as s:
                                try:
                                    results = run_ta_analysis(
                                        t,
                                        str(date.today() - timedelta(days=1)),
                                        write_progress=st.write,
                                    )
                                    save_analysis_to_position(t, results)
                                    s.update(
                                        label=f"{t} updated — {results['signal']}",
                                        state="complete", expanded=False,
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    s.update(label=f"{t} failed", state="error")
                                    st.error(str(exc))

                # Expandable: last analysis reports
                if pos.get("final_trade_decision"):
                    with st.expander("View last analysis reports"):
                        render_analysis_results({
                            "ticker": t,
                            "trade_date": pos.get("last_analysis_date", ""),
                            "signal": signal,
                            "fundamentals_report": pos.get("fundamentals_report", ""),
                            "news_report": pos.get("news_report", ""),
                            "sentiment_report": pos.get("sentiment_report", ""),
                            "invest_judge_decision": pos.get("invest_judge_decision", ""),
                            "final_trade_decision": pos.get("final_trade_decision", ""),
                            "risk_judge_decision": pos.get("risk_judge_decision", ""),
                        })

    # ── Add position manually ─────────────────────────────────────────────────

    st.divider()
    with st.expander("➕ Add a Position Manually"):
        st.markdown(
            "Add a holding without running an analysis first. "
            "You can update it later with the **Update** button."
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            manual_ticker = st.text_input(
                "Ticker", placeholder="e.g. AAPL", key="manual_ticker"
            ).upper().strip()
        with mc2:
            manual_shares = st.number_input(
                "Shares", min_value=0.0001, value=1.0, step=0.5, key="manual_shares"
            )
        with mc3:
            manual_price = st.number_input(
                "Buy price ($)", min_value=0.01, value=100.0, step=1.0,
                key="manual_price"
            )
        with mc4:
            manual_date = st.date_input(
                "Date purchased", value=date.today() - timedelta(days=1),
                max_value=date.today(), key="manual_date"
            )
        if st.button("Add Position", key="btn_manual_add"):
            if not manual_ticker:
                st.error("Please enter a ticker symbol.")
            else:
                upsert_position(manual_ticker, manual_shares, manual_price,
                                str(manual_date))
                st.success(f"**{manual_ticker}** added to your portfolio.")
                st.rerun()

    st.divider()
    st.caption(
        "⚠️ AI-generated for educational purposes only. Not financial advice. "
        "Consult a qualified financial advisor before investing."
    )

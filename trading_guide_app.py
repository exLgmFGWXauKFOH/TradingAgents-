"""
Trading Guide — A beginner-friendly stock analysis web app.
Powered by TradingAgents multi-agent AI framework with Anthropic Claude.

Run with:
    streamlit run trading_guide_app.py
"""

import os
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

# ── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Trading Guide",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📘 How It Works")
    st.markdown(
        """
        Enter a stock ticker and a date. Our AI agents will analyze the
        company and give you a plain-English trading guide.

        **What the AI examines:**

        - 📊 **Company Fundamentals** — earnings, revenue, debt, valuation
        - 📰 **News & Sentiment** — recent headlines and public mood
        - 🤖 **Bull vs. Bear Debate** — one AI argues to buy, another to sell,
          a judge decides who's right
        - ⚖️ **Risk Assessment** — is now a safe time to act?

        **Result:** A clear **BUY / HOLD / SELL** recommendation with reasoning.

        ---
        **Tips for beginners:**
        - Use the ticker symbol, not the company name (e.g. `AAPL` not `Apple`)
        - Use a past date — real-time data may have a 1-day delay
        - Analysis takes 2–5 minutes to complete

        ---
        ⚠️ *For educational purposes only. Not financial advice.*
        """
    )

    st.divider()
    st.markdown("**Quick reference: Common tickers**")
    st.markdown(
        """
        | Company | Ticker |
        |---------|--------|
        | Apple | AAPL |
        | Tesla | TSLA |
        | Microsoft | MSFT |
        | Nvidia | NVDA |
        | Amazon | AMZN |
        | Google | GOOGL |
        | Meta | META |
        """
    )

# ── Main header ──────────────────────────────────────────────────────────────

st.title("📈 AI Trading Guide")
st.markdown(
    "Enter a stock and a date below. AI agents will research the company and "
    "deliver a structured trading guide written for beginners."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────

col_ticker, col_date, col_btn = st.columns([2, 2, 1])

with col_ticker:
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="The short code for the stock (e.g. AAPL for Apple, TSLA for Tesla).",
        placeholder="e.g. AAPL",
    ).upper().strip()

with col_date:
    yesterday = date.today() - timedelta(days=1)
    trade_date = st.date_input(
        "Analysis Date",
        value=yesterday,
        max_value=yesterday,
        help="The date to run analysis for. Use yesterday or earlier for best data coverage.",
    )

with col_btn:
    st.write("")  # vertical alignment spacer
    st.write("")
    run_clicked = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ── Run analysis ──────────────────────────────────────────────────────────────

if run_clicked:
    if not ticker:
        st.error("Please enter a stock ticker symbol.")
        st.stop()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "**ANTHROPIC_API_KEY not found.**  \n"
            "Add your key to a `.env` file in the project root:  \n"
            "`ANTHROPIC_API_KEY=sk-ant-...`"
        )
        st.stop()

    # Build config — Claude models, yfinance for data (no extra API keys needed)
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-opus-4-6"
    config["quick_think_llm"] = "claude-sonnet-4-6"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    # Live progress via st.status
    with st.status(
        f"Researching **{ticker}** for {trade_date}...", expanded=True
    ) as status:
        try:
            st.write("Initializing AI agents (this may take a moment)...")

            ta = TradingAgentsGraph(
                selected_analysts=["fundamentals", "news", "social"],
                config=config,
            )

            st.write("Agents ready. Starting research...")

            init_state = ta.propagator.create_initial_state(ticker, str(trade_date))
            graph_args = ta.propagator.get_graph_args()

            # Track which milestones have been announced
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
                    st.write("✅ Fundamentals analysis complete")
                    milestones["fundamentals_report"] = True

                if chunk.get("news_report") and not milestones["news_report"]:
                    st.write("✅ News analysis complete")
                    milestones["news_report"] = True

                if chunk.get("sentiment_report") and not milestones["sentiment_report"]:
                    st.write("✅ Sentiment analysis complete")
                    milestones["sentiment_report"] = True

                invest_state = chunk.get("investment_debate_state") or {}
                if invest_state.get("judge_decision") and not milestones["invest_judge"]:
                    st.write("✅ Bull vs. Bear debate complete")
                    milestones["invest_judge"] = True

                if chunk.get("final_trade_decision") and not milestones["final_trade_decision"]:
                    st.write("✅ Final recommendation ready")
                    milestones["final_trade_decision"] = True

                last_chunk = chunk

            # Extract the processed signal (BUY / SELL / HOLD / etc.)
            signal = ta.process_signal(last_chunk["final_trade_decision"])

            # Store results in session state so they survive Streamlit reruns
            # (e.g. when the user clicks between tabs)
            risk_state = last_chunk.get("risk_debate_state") or {}
            invest_state = last_chunk.get("investment_debate_state") or {}

            st.session_state["guide_results"] = {
                "ticker": ticker,
                "trade_date": str(trade_date),
                "signal": signal.strip().upper(),
                "fundamentals_report": last_chunk.get("fundamentals_report", ""),
                "news_report": last_chunk.get("news_report", ""),
                "sentiment_report": last_chunk.get("sentiment_report", ""),
                "invest_judge_decision": invest_state.get("judge_decision", ""),
                "final_trade_decision": last_chunk.get("final_trade_decision", ""),
                "risk_judge_decision": risk_state.get("judge_decision", ""),
            }

            status.update(label="Research complete!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"**Error during analysis:** {exc}")
            st.stop()

# ── Display results ───────────────────────────────────────────────────────────

if "guide_results" in st.session_state:
    r = st.session_state["guide_results"]

    st.divider()

    # ── Recommendation banner ─────────────────────────────────────────────────

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

    bg, border, text = SIGNAL_STYLE.get(r["signal"], ("#e2e3e5", "#6c757d", "#383d41"))
    icon = SIGNAL_ICON.get(r["signal"], "⚪")

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:2px solid {border};
            border-radius:12px;
            padding:24px;
            text-align:center;
            margin-bottom:24px;
        ">
            <div style="font-size:2.8rem; font-weight:800; color:{text}; margin:0;">
                {icon}&nbsp;{r['signal']}
            </div>
            <div style="font-size:1.1rem; color:{text}; margin-top:6px;">
                AI recommendation for&nbsp;<strong>{r['ticker']}</strong>
                &nbsp;as of&nbsp;<strong>{r['trade_date']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Beginner signal explainer ──────────────────────────────────────────────

    SIGNAL_EXPLAIN = {
        "BUY": (
            "The AI agents believe this stock is currently **undervalued or has strong "
            "upward momentum**. Based on the research, this could be a good time to "
            "consider buying — but always do your own research and invest only what "
            "you can afford to lose."
        ),
        "OVERWEIGHT": (
            "The AI agents recommend holding **more of this stock** than its market "
            "weighting. Think of it as a soft 'Buy' — the stock looks attractive relative "
            "to the market."
        ),
        "HOLD": (
            "The AI agents see **roughly balanced** reasons to buy and sell. The stock "
            "is not screaming cheap or expensive right now. If you already own it, "
            "consider keeping it. If you don't, there may be better opportunities elsewhere."
        ),
        "UNDERWEIGHT": (
            "The AI agents suggest holding **less of this stock** than its market weighting. "
            "Think of it as a soft 'Sell' — some caution is warranted."
        ),
        "SELL": (
            "The AI agents see **significant risks or overvaluation**. Based on the research, "
            "this may not be a good time to hold the stock. If you own it, consider reviewing "
            "your position."
        ),
    }

    with st.expander("What does this mean for a beginner?", expanded=True):
        st.markdown(SIGNAL_EXPLAIN.get(r["signal"], "See the tabs below for the full analysis."))

    st.divider()

    # ── Report tabs ────────────────────────────────────────────────────────────

    tab_fund, tab_news, tab_rec = st.tabs([
        "📊 Company Fundamentals",
        "📰 News & Sentiment",
        "🤖 Full AI Recommendation",
    ])

    with tab_fund:
        st.subheader("Company Fundamentals")
        st.markdown(
            "_Fundamentals describe a company's financial health — revenue, profits, "
            "debt, and how the business is performing. Strong fundamentals usually "
            "indicate a well-run company._"
        )
        st.divider()
        if r["fundamentals_report"]:
            st.markdown(r["fundamentals_report"])
        else:
            st.info("No fundamentals report was generated for this analysis.")

    with tab_news:
        st.subheader("News & Sentiment")
        st.markdown(
            "_News shows what's happening around the company right now. Sentiment "
            "reflects the general mood of investors and the public — positive, "
            "negative, or neutral._"
        )
        st.divider()

        if r["news_report"]:
            st.markdown("#### News Report")
            st.markdown(r["news_report"])
        else:
            st.info("No news report was generated for this analysis.")

        if r["sentiment_report"]:
            st.divider()
            st.markdown("#### Sentiment Report")
            st.markdown(r["sentiment_report"])
        elif not r["news_report"]:
            st.info("No sentiment data was generated for this analysis.")

    with tab_rec:
        st.subheader("AI Recommendation")
        st.markdown(
            "_This is the final decision produced by the AI agents after weighing all "
            "the evidence — fundamentals, news, a bull vs. bear debate, and a risk "
            "management review._"
        )
        st.divider()

        if r["final_trade_decision"]:
            st.markdown(r["final_trade_decision"])
        else:
            st.info("No final recommendation was generated.")

        if r["invest_judge_decision"]:
            with st.expander("Bull vs. Bear Debate — Judge's Decision"):
                st.markdown(
                    "_One AI agent argued for buying (bull), one argued against (bear). "
                    "Here's how the judge ruled:_"
                )
                st.markdown(r["invest_judge_decision"])

        if r["risk_judge_decision"]:
            with st.expander("Risk Assessment — Risk Team's Verdict"):
                st.markdown(
                    "_The risk management team evaluated whether the trade recommendation "
                    "is appropriate given current market conditions._"
                )
                st.markdown(r["risk_judge_decision"])

    # ── Footer disclaimer ──────────────────────────────────────────────────────

    st.divider()
    st.caption(
        "⚠️ This trading guide is generated by AI for **educational purposes only** "
        "and does not constitute financial advice. Past performance is not indicative "
        "of future results. Always consult a qualified financial advisor before making "
        "investment decisions."
    )

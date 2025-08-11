import os, time, threading, asyncio, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Make sure we can import our local modules when running on Streamlit Cloud
sys.path.append("src")

# --------- SAFE IMPORTS (flow optional) ----------
from msv_stream_bus import start_stream_bus, get_df, store
try:
    from msv_flow import start_flow_monitor
    HAVE_FLOW=True
except Exception:
    HAVE_FLOW=False
    async def start_flow_monitor(symbols, flow_state):  # no-op fallback
        while True:
            await asyncio.sleep(5)

from msv_beast_core import beast_from_df

# ---------------- CONFIG ----------------
SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
DEFAULT_THRESH = int(os.getenv("THRESH","75"))
REFRESH_MS = 1500  # soft refresh

flow_state = defaultdict(dict)

# ---------------- BACKGROUND TASKS ----------------
def _bg_runner(loop):
    asyncio.set_event_loop(loop)
    tasks = [loop.create_task(start_stream_bus(SYMS))]
    tasks.append(loop.create_task(start_flow_monitor(SYMS, flow_state)))
    loop.run_forever()

def ensure_background():
    if "bg_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state["bg_loop"] = loop
        t = threading.Thread(target=_bg_runner, args=(loop,), daemon=True)
        t.start()

# ---------------- HELPERS ----------------
def pct_change(cur, past):
    try:
        if cur is None or past in (None, 0): return 0.0
        return (cur/past - 1.0)*100.0
    except Exception:
        return 0.0

def last_price(sym):
    df = get_df(sym, "1m", 200)
    if df.empty: return None
    return float(df["close"].iloc[-1])

def hist_price(sym, seconds=60):
    df = get_df(sym, "1m", 200)
    if df.empty: return None
    if seconds <= 60 and len(df)>=2:
        return float(df["close"].iloc[-2])
    mins = max(1, int(seconds//60))
    if len(df) > mins:
        return float(df["close"].iloc[-(mins+1)])
    return float(df["close"].iloc[0])

def nice_num(x):
    if x is None: return "-"
    return f"{x:,.0f}" if x>=1000 else f"{x:,.2f}"

def render_symbol_card(sym, thresh):
    # Prices & changes
    p_last = last_price(sym)
    p_1m   = hist_price(sym, 60)
    p_5m   = hist_price(sym, 300)
    ch1 = pct_change(p_last, p_1m)
    ch5 = pct_change(p_last, p_5m)

    # Flow (may be missing early)
    fs = flow_state.get(sym, {})
    obi = fs.get("obi_bbo", 0.0)
    cvd5= fs.get("cvd_5m", 0.0)

    # 5m report (from in-memory stream)
    df5 = get_df(sym, "5m", 800)
    rep = beast_from_df(sym, df5)
    score = rep.get("confluence",{}).get("score", 0)
    sig   = rep.get("signal",{})
    dirn = sig.get("direction")
    entry, stop, tp1, tp2 = sig.get("entry"), sig.get("stop"), sig.get("tp1"), sig.get("tp2")
    orb = rep.get("orb",{})

    # ---- TILE ROW ----
    top = st.container()
    cA,cB,cC,cD,cE,cF = top.columns([1.1,1,1,1,1,1.2])

    with cA:
        st.markdown(f"### **{sym}**")
        st.caption("Live • 1m/5m")
    cB.metric("Last", nice_num(p_last))
    cC.metric("1m %", f"{ch1:+.2f}%")
    cD.metric("5m %", f"{ch5:+.2f}%")
    cE.metric("OBI %", f"{obi:+.2f}" if HAVE_FLOW else "—")
    cF.metric("CVD 5m", f"{cvd5:+.1f}" if HAVE_FLOW else "—")

    # ---- SIGNAL ROW ----
    row2 = st.container()
    t1,t2,t3,t4,t5 = row2.columns([1,1.2,1.2,1.2,1.2])
    color = "green" if score>=thresh else ("orange" if score>=60 else "gray")
    t1.markdown(f"**Confluence**: :{color}[{score}]")
    sig_txt = "-" if not dirn else (":green[LONG]" if dirn=="long" else ":red[SHORT]")
    t2.markdown(f"**Signal**: {sig_txt}")
    t3.markdown(f"**Entry**: :red[{nice_num(entry)}]") if entry else t3.markdown("**Entry**: -")
    t4.markdown(f"**Stop**: :violet[{nice_num(stop)}]") if stop else t4.markdown("**Stop**: -")
    if tp1:
        t5.markdown(f"**TP1/TP2**: :cyan[{nice_num(tp1)}] / :cyan[{nice_num(tp2)}]")
    else:
        t5.markdown("**TP1/TP2**: -")

    # ---- CHART ----
    card = st.container()
    if df5.empty:
        card.info("Waiting for live 5m bars…")
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df5.index, open=df5["open"], high=df5["high"], low=df5["low"], close=df5["close"],
            name="5m"
        ))
        if orb.get("high") and orb.get("low"):
            fig.add_hline(y=orb["high"], line_color="green", line_dash="dot", annotation_text="ORB High")
            fig.add_hline(y=orb["low"],  line_color="red",   line_dash="dot", annotation_text="ORB Low")
        if entry: fig.add_hline(y=entry, line_color="red",    annotation_text="Entry")
        if stop:  fig.add_hline(y=stop,  line_color="magenta",annotation_text="Stop")
        if tp1:   fig.add_hline(y=tp1,   line_color="cyan",   annotation_text="TP1")
        if tp2:   fig.add_hline(y=tp2,   line_color="cyan",   annotation_text="TP2")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ---------------- PAGE ----------------
st.set_page_config(page_title="MSV Live Crypto", layout="wide", page_icon="📊", initial_sidebar_state="collapsed")
ensure_background()

with st.sidebar:
    st.header("MSV Live Crypto")
    thresh = st.slider("Alert Threshold", min_value=50, max_value=95, value=DEFAULT_THRESH, step=1)
    st.caption("Symbols: " + ", ".join(SYMS))
    st.caption("Colors: Entry (red), Stop (violet), TPs (cyan). Score turns green at/above threshold.")
    auto = st.toggle("Auto-refresh", value=True)
    ms   = st.select_slider("Refresh ms", options=[500,1000,1500,2000,3000], value=REFRESH_MS)

st.title("MSV Live — BTC / ETH / BNB / SOL")
st.caption("Live confluence: ORB + PO3 + BOS + VWAP + BB/RSI/StochRSI + Volume + Flow (OBI/CVD).")

for s in SYMS:
    render_symbol_card(s, thresh)

# simple refresh loop (no deprecated APIs)
if auto:
    time.sleep(ms/1000.0)
    st.rerun()

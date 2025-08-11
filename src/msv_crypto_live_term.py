import os, sys, json, time, math, asyncio, requests, websockets, traceback
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict

# rich UI
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

# make sure we can import our beast logic
sys.path.append("src")
from msv_crypto_beast import build_report  # uses Binance REST under the hood

SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
THRESH = float(os.getenv("THRESH","75"))        # alert when confluence >= THRESH
SHOW_ROWS = len(SYMS)

USE_GPT = os.getenv("ADVICE_GPT","0") == "1"
GPT_MODEL = os.getenv("GPT_MODEL","gpt-5")      # per OpenAI docs, see API ref
GPT_MAXTOK = int(os.getenv("GPT_MAXTOK","160"))

# simple in-memory state
last_tick = {s: None for s in SYMS}
last_trade_ts = {s: 0 for s in SYMS}
price_buf = {s: deque(maxlen=600) for s in SYMS}  # last 10 minutes of per-second prices
last_signal = {s: None for s in SYMS}
last_score = {s: 0 for s in SYMS}
last_report = {s: None for s in SYMS}

console = Console()

def pct(a,b):
    try:
        return (a/b - 1.0) * 100.0
    except Exception:
        return 0.0

def color_num(val, fmt="{:.2f}"):
    t = fmt.format(val)
    if val > 0:   return f"[green]{t}[/green]"
    if val < 0:   return f"[red]{t}[/red]"
    return t

def format_price(p):
    if p is None: return "-"
    if p>=1000: return f"{p:,.0f}"
    if p>=100:  return f"{p:,.2f}"
    return f"{p:,.4f}"

def build_table():
    tbl = Table(title="MSV Crypto LIVE — BTC/ETH/BNB/SOL", box=box.SIMPLE_HEAVY)
    for col in ["Symbol","Last","1m %","5m %","Confluence","Signal","Entry","Stop","TP1","TP2"]:
        tbl.add_column(col, justify="right" if col!="Symbol" else "left")
    for s in SYMS:
        tick = last_tick.get(s) or {}
        last = tick.get("last")
        # 1m/5m changes
        now = time.time()
        def pct_win(seconds):
            # find price approx seconds ago
            dq = price_buf[s]
            if len(dq)<2: return 0.0
            target = now - seconds
            base = None
            for ts,pr in reversed(dq):
                if ts <= target: 
                    base = pr; break
            if base is None: base = dq[0][1]
            return pct(last, base) if (last and base) else 0.0

        ch1 = pct_win(60)
        ch5 = pct_win(300)

        score = last_score.get(s,0)
        sig   = last_signal.get(s) or {}
        dirn  = sig.get("direction")
        entry = sig.get("entry")
        stop  = sig.get("stop")
        tp1   = sig.get("tp1")
        tp2   = sig.get("tp2")

        # colors
        score_txt = f"[bold]{score:.0f}[/bold]"
        if score >= 85: score_txt = f"[bold bright_green]{score:.0f}[/bold bright_green]"
        elif score >= THRESH: score_txt = f"[bold green]{score:.0f}[/bold green]"
        elif score >= 60: score_txt = f"[yellow]{score:.0f}[/yellow]"

        sig_txt  = "-"
        if dirn == "long":  sig_txt = "[green]LONG[/green]"
        if dirn == "short": sig_txt = "[red]SHORT[/red]"

        entry_txt = f"[red]{format_price(entry)}[/red]" if entry else "-"
        stop_txt  = f"[magenta]{format_price(stop)}[/magenta]" if stop else "-"
        tp1_txt   = f"[cyan]{format_price(tp1)}[/cyan]" if tp1 else "-"
        tp2_txt   = f"[cyan]{format_price(tp2)}[/cyan]" if tp2 else "-"

        tbl.add_row(
            f"[bold]{s}[/bold]",
            format_price(last),
            color_num(ch1),
            color_num(ch5),
            score_txt,
            sig_txt,
            entry_txt,
            stop_txt,
            tp1_txt,
            tp2_txt
        )
    return tbl

async def advise_with_gpt(report):
    try:
        from openai import OpenAI
        client = OpenAI()  # uses OPENAI_API_KEY env
        sig = report.get("signal") or {}
        msg = {
            "role":"user",
            "content":(
                "Act as a cautious trading coach. Given this crypto state, offer 1–2 lines of risk-aware advice. "
                "Do NOT make promises or position sizing. Prefer 'consider/warning' language.\n\n"
                + json.dumps(report, indent=2)
            )
        }
        # Chat Completions per OpenAI API docs
        out = client.chat.completions.create(
            model=os.getenv("GPT_MODEL","gpt-5"),
            messages=[{"role":"system","content":"You are a concise trading coach."}, msg],
            max_tokens=int(os.getenv("GPT_MAXTOK","120"))
        )
        return out.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT advice unavailable: {e})"

async def handle_stream():
    # combined WS: tickers + 5m klines
    stream = "wss://stream.binance.com:9443/stream?streams=" + "/".join(
        [f"{s.lower()}@ticker" for s in SYMS] + [f"{s.lower()}@kline_5m" for s in SYMS]
    )
    console.print(f"[bold]Connected[/bold] to Binance streams for: {', '.join(SYMS)}  — alerts when score ≥ {THRESH}\n")

    # live UI loop
    with Live(build_table(), console=console, refresh_per_second=8, screen=False) as live:
        async with websockets.connect(stream, ping_interval=20) as ws:
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                data = msg.get("data", {})
                stype = data.get("e") or ("kline" if "k" in data else None)

                if stype == "24hrTicker":
                    s = data["s"]
                    last = float(data["c"])
                    last_tick[s] = {"last": last, "bid": float(data["b"]), "ask": float(data["a"])}
                    price_buf[s].append((time.time(), last))
                    live.update(build_table())
                    continue

                if stype == "kline":
                    k = data["k"]; s = k["s"]
                    last = float(k["c"])
                    last_tick[s] = {"last": last}
                    price_buf[s].append((time.time(), last))
                    # only act on candle close
                    if k["x"]:
                        try:
                            rep = build_report(s)  # re-run beast for that symbol
                            last_report[s] = rep
                            last_score[s] = rep["confluence"]["score"]
                            last_signal[s] = rep["signal"]
                            live.update(build_table())
                            # alert if threshold
                            if rep["signal"]["direction"] and rep["confluence"]["score"] >= THRESH:
                                line = (f"{s} score {rep['confluence']['score']} | "
                                        f"{rep['signal']['direction'].upper()} entry {rep['signal']['entry']} "
                                        f"stop {rep['signal']['stop']} tp1 {rep['signal']['tp1']} tp2 {rep['signal']['tp2']}")
                                console.print(f"[bold red]ENTRY[/bold red] {line}\a")
                                # GPT advice (optional)
                                if USE_GPT:
                                    advice = await advise_with_gpt(rep)
                                    console.print(f"[italic yellow]GPT-5:[/italic yellow] {advice}")
                        except Exception as e:
                            console.print(f"[red]Error recomputing {s}: {e}[/red]")
                            traceback.print_exc()
                    live.update(build_table())
                    continue

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(handle_stream())
    except KeyboardInterrupt:
        print("\nStopped")

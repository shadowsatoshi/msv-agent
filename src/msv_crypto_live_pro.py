import os, json, time, asyncio, websockets, requests, sys
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except Exception:
    pass
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
from collections import defaultdict
from msv_flow import start_flow_monitor

sys.path.append("src")
from msv_crypto_beast import build_report  # legacy (kept)
from msv_stream_bus import start_stream_bus, get_df
from msv_beast_core import beast_from_df
from msv_mtf_patterns import htf_snapshot

SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
HTF_SHOW = ["10m","45m","2h"]  # which HTFs to display as columns
THRESH = float(os.getenv("THRESH","75"))
HTF_LIST = ["10m","15m","30m","45m","1h","2h"]   # add/remove as you like
USE_GPT = os.getenv("ADVICE_GPT","0")=="1"

console = Console()
flow_state = defaultdict(dict)
last = {s: {"price": None, "score": 0, "signal": None, "htf": {}} for s in SYMS}
last_refresh = {s: 0 for s in SYMS}

def color_dir(x):
    return "[green]UP[/green]" if x=="UP" else "[red]DOWN[/red]" if x=="DOWN" else "[yellow]NEUTRAL[/yellow]"

def build_table():
    tbl=Table(title="MSV Crypto LIVE PRO — 5m confluence + HTF bias + patterns", box=box.SIMPLE_HEAVY)
    tbl.add_column("Symbol")
    tbl.add_column("Last", justify="right")
    tbl.add_column("OBI%", justify="right")
    tbl.add_column("CVD5m", justify="right")
    tbl.add_column("Score(5m)", justify="center")
    for h in HTF_SHOW:
        tbl.add_column(h, justify="center")
    tbl.add_column("Pattern", justify="left")
    tbl.add_column("Signal", justify="center")
    tbl.add_column("Entry", justify="right")
    tbl.add_column("Stop", justify="right")
    tbl.add_column("TP1", justify="right")
    tbl.add_column("TP2", justify="right")
    for s in SYMS:
        st=last[s]
        p=st["price"]; price = "-" if p is None else (f"{p:,.0f}" if p>=1000 else f"{p:,.2f}")
        fs = flow_state.get(s, {})
        obi = fs.get("obi_bbo", 0.0)
        cvd5 = fs.get("cvd_5m", 0.0)
        def cnum(v):
            txt=f"{v:+.1f}" if abs(v)>=1 else f"{v:+.2f}"
            return f"[green]{txt}[/green]" if v>0 else (f"[red]{txt}[/red]" if v<0 else txt)
        obi_txt = cnum(obi)
        cvd_txt = cnum(cvd5)
        sc=st["score"]
        sc_txt = f"[bold]{sc:.0f}[/bold]"
        if sc>=85: sc_txt=f"[bold bright_green]{sc:.0f}[/bold bright_green]"
        elif sc>=THRESH: sc_txt=f"[bold green]{sc:.0f}[/bold green]"
        elif sc>=60: sc_txt=f"[yellow]{sc:.0f}[/yellow]"
        # htf
        htf_cells=[]
        for hh in HTF_SHOW:
            rr=st["htf"].get(hh,{})
            htf_cells.append(color_dir(rr.get("bias","-")) if rr else "-")
        # pattern
        pat=None
        for k in ["1h","30m","15m","4h","1d"]:
            if st["htf"].get(k,{}).get("pattern"):
                pat=st["htf"][k]["pattern"]; break
        if pat:
            name = pat.get('pattern')
            diru = pat.get('direction')
            up = (diru=='up')
            if name=='H&S': pat_txt=f"[red]H&S↓[/red] brk @ {pat['break_time']}"
            elif name=='iH&S': pat_txt=f"[green]iH&S↑[/green] brk @ {pat['break_time']}"
            elif name=='DT': pat_txt=f"[red]DT↓[/red] brk @ {pat['break_time']}"
            elif name=='DB': pat_txt=f"[green]DB↑[/green] brk @ {pat['break_time']}"
            elif name=='TRI': pat_txt=(f"[green]TRI↑[/green]" if up else f"[red]TRI↓[/red]") + f" {pat.get('sub','')} brk @ {pat['break_time']}"
            elif name=='WEDGE': pat_txt=(f"[green]WEDGE↑[/green]" if up else f"[red]WEDGE↓[/red]") + f" {pat.get('sub','')} brk @ {pat['break_time']}"
            elif name=='FLAG': pat_txt=(f"[green]FLAG↑[/green]" if up else f"[red]FLAG↓[/red]") + f" {pat.get('sub','')} brk @ {pat['break_time']}"
            else: pat_txt='-'
        else:
            pat_txt="-"
        sig=st["signal"] or {}
        sdir=sig.get("direction")
        sig_txt = "[green]LONG[/green]" if sdir=="long" else "[red]SHORT[/red]" if sdir=="short" else "-"
        entry = f"[red]{sig.get('entry','-')}[/red]" if sdir else "-"
        stop  = f"[magenta]{sig.get('stop','-')}[/magenta]" if sdir else "-"
        tp1   = f"[cyan]{sig.get('tp1','-')}[/cyan]" if sdir else "-"
        tp2   = f"[cyan]{sig.get('tp2','-')}[/cyan]" if sdir else "-"
        tbl.add_row(s, price, obi_txt, cvd_txt, sc_txt, *htf_cells, pat_txt, sig_txt, entry, stop, tp1, tp2)
    return tbl

async def compute_htf(symbol):
    out={}
    for itv in HTF_LIST:
        try:
            out[itv]=htf_snapshot(symbol, itv, days=120 if itv in ("1d","1w","1M") else 60)
        except Exception as e:
            out[itv]={"interval":itv,"error":str(e)}
    return out

async def run():
    bus_task = asyncio.create_task(start_stream_bus(SYMS))
    stream="wss://stream.binance.com:9443/stream?streams=" + "/".join(
        [f"{s.lower()}@ticker" for s in SYMS] + [f"{s.lower()}@kline_5m" for s in SYMS]
    )
    console.print(f"[bold]Connected[/bold] to Binance — symbols: {', '.join(SYMS)}  | alert ≥ {THRESH}\n")
    # initial HTF
    for s in SYMS:
        last[s]["htf"]=await compute_htf(s)
        # seed price via REST so table shows immediately
        try:
            import requests
            pr=requests.get('https://api.binance.com/api/v3/ticker/price',params={'symbol':s},timeout=8).json()
            last[s]['price']=float(pr['price'])
        except: pass
        # seed score/signal once at start
        try:
            df5 = get_df(s, "5m", lookback=2000)

            rep = beast_from_df(s, df5)
            last[s]['score']=rep['confluence']['score']
            last[s]['signal']=rep['signal']
        except: pass
    flow_task = asyncio.create_task(start_flow_monitor(SYMS, flow_state))
    with Live(build_table(), console=console, refresh_per_second=4, screen=False) as live:
        async with websockets.connect(stream, ping_interval=20) as ws:
            while True:
                msg=json.loads(await ws.recv())
                data=msg.get("data",{})
                if "e" in data and data["e"]=="24hrTicker":
                    s=data["s"]; last[s]["price"]=float(data["c"])
                    live.update(build_table()); continue
                if "k" in data:  # kline
                    k=data["k"]; s=k["s"]; last[s]["price"]=float(k["c"])
                    # refresh HTF every 5m close or every 10 minutes
                    now=time.time()
                    if now - last_refresh[s] > 900:
                        last[s]["htf"]=await compute_htf(s); last_refresh[s]=now
                    if k["x"]:  # candle closed
                        try:
                            df5 = get_df(s, "5m", lookback=2000)

                            rep = beast_from_df(s, df5)
                            last[s]["score"]=rep["confluence"]["score"]
                            last[s]["signal"]=rep["signal"]
                            live.update(build_table())
                            if rep["signal"]["direction"] and rep["confluence"]["score"]>=THRESH:
                                line=(f"{s} score {rep['confluence']['score']} | "
                                      f"{rep['signal']['direction'].upper()} entry {rep['signal']['entry']} "
                                      f"stop {rep['signal']['stop']} tp1 {rep['signal']['tp1']} tp2 {rep['signal']['tp2']}")
                                agree = (flow_state[s].get('obi_bbo',0)>0 and rep['signal']['direction']=='long') or (flow_state[s].get('obi_bbo',0)<0 and rep['signal']['direction']=='short')
                                agree = agree and (((flow_state[s].get('cvd_5m',0)>0) if rep['signal']['direction']=='long' else (flow_state[s].get('cvd_5m',0)<0)))
                                tag = ' [green][flow✓][/green]' if agree else ''
                                console.print(f"[bold red]ENTRY[/bold red] {line}{tag}\a")
                        except Exception as e:
                            console.print(f"[red]{s} compute error: {e}[/red]")
                    live.update(build_table())

if __name__=="__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Stopped")

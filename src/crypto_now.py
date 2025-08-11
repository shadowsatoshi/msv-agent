import asyncio, json, sys, websockets, time
sym = (sys.argv[1] if len(sys.argv)>1 else "BTCUSDT").lower()
url = f"wss://stream.binance.com:9443/ws/{sym}@ticker"
async def run():
    async with websockets.connect(url, ping_interval=20) as ws:
        print(f"Connected to Binance for {sym.upper()} (Ctrl+C to quit)")
        while True:
            msg = json.loads(await ws.recv())
            last = float(msg["c"]); bid=float(msg["b"]); ask=float(msg["a"])
            chg = float(msg["p"]); chg_pct = float(msg["P"])
            vol = float(msg["v"])
            print(f"{time.strftime('%H:%M:%S')}  {sym.upper()}  last {last:.2f}  bid {bid:.2f}  ask {ask:.2f}  24h {chg:+.2f} ({chg_pct:+.2f}%)  vol {vol:.0f}")
asyncio.run(run())

import asyncio, json, time, websockets
from collections import defaultdict, deque

# state[symbol] -> {'obi_bbo': %, 'spread': %, 'cvd_1m': float, 'cvd_5m': float, 'ts': epoch}
async def start_flow_monitor(symbols, state):
    streams = "/".join([f"{s.lower()}@bookTicker" for s in symbols] +
                       [f"{s.lower()}@aggTrade"   for s in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    # per-symbol rolling windows for CVD
    cvd1 = {s: deque() for s in symbols}   # (ts, +/- qty) last 60s
    cvd5 = {s: deque() for s in symbols}   # last 300s
    async with websockets.connect(url, ping_interval=20) as ws:
        while True:
            msg = json.loads(await ws.recv())
            d = msg.get("data", {})
            et = d.get("e")
            now = time.time()
            if et == "bookTicker":
                s = d["s"]
                bid, ask = float(d["b"]), float(d["a"])
                bidq, askq = float(d["B"]), float(d["A"])
                den = bidq + askq
                obi = ((bidq - askq) / den * 100.0) if den > 0 else 0.0
                state[s]["obi_bbo"] = obi
                state[s]["spread"]  = ((ask - bid) / bid * 100.0) if bid > 0 else 0.0
                state[s]["ts"] = now
            elif et == "aggTrade":
                s = d["s"]
                qty = float(d["q"])
                # 'm' True => buyer is maker => sell-initiated
                delta = -qty if d.get("m", False) else qty
                cvd1[s].append((now, delta)); cvd5[s].append((now, delta))
                cut1, cut5 = now - 60, now - 300
                while cvd1[s] and cvd1[s][0][0] < cut1: cvd1[s].popleft()
                while cvd5[s] and cvd5[s][0][0] < cut5: cvd5[s].popleft()
                state[s]["cvd_1m"] = sum(v for _, v in cvd1[s])
                state[s]["cvd_5m"] = sum(v for _, v in cvd5[s])
                state[s]["ts"] = now

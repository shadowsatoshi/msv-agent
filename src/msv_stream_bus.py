import asyncio, json, time, websockets
from collections import deque, defaultdict
import pandas as pd

# store[sym]['1m'|'5m'] -> deque of (t,o,h,l,c,v) with UTC pandas Timestamps
store = defaultdict(lambda: {'1m': deque(maxlen=8000), '5m': deque(maxlen=4000)})

def _upd(queue, k, closed):
    t = pd.to_datetime(k['T'], unit='ms', utc=True)
    o,h,l,c,v = map(float, (k['o'],k['h'],k['l'],k['c'],k['v']))
    if closed:
        queue.append((t,o,h,l,c,v))
    else:
        if queue and queue[-1][0] == t:
            queue[-1] = (t,o,h,l,c,v)

async def start_stream_bus(symbols):
    streams = "/".join([f"{s.lower()}@kline_1m" for s in symbols] + [f"{s.lower()}@kline_5m" for s in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    async with websockets.connect(url, ping_interval=20) as ws:
        while True:
            msg = json.loads(await ws.recv())
            d = msg.get("data", {})
            if "k" not in d: continue
            k = d["k"]; s = k["s"]
            itv = k["i"]
            closed = k["x"]
            if itv == "1m":
                _upd(store[s]['1m'], k, closed)
            elif itv == "5m":
                _upd(store[s]['5m'], k, closed)

def get_df(symbol, tf="5m", lookback=2000):
    arr = list(store[symbol][tf])[-lookback:]
    if not arr: return pd.DataFrame()
    idx = [a[0] for a in arr]
    df = pd.DataFrame(arr, columns=["time","open","high","low","close","volume"]).set_index("time")
    df.index.name = "time_utc"
    return df

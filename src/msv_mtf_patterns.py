import time, requests, json, math
import pandas as pd, numpy as np

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000

def fetch_klines(symbol, interval, start_ms, end_ms):
    out=[]; cur=start_ms
    while cur<end_ms:
        p={"symbol":symbol,"interval":interval,"limit":MAX_LIMIT,"startTime":cur,"endTime":end_ms}
        r=requests.get(BINANCE_URL, params=p, timeout=15); r.raise_for_status()
        batch=r.json()
        if not batch: break
        out.extend(batch)
        nxt=batch[-1][6]+1
        if nxt<=cur: break
        cur=nxt
        if len(batch)<MAX_LIMIT: break
    return out

def load_df(symbol, interval, days=60):
    now=int(time.time()*1000); start=now - days*24*60*60*1000
    raw=fetch_klines(symbol, interval, start, now)
    if not raw: return pd.DataFrame()
    cols=["ot","open","high","low","close","volume","ct","qv","trades","tb","tq","ig"]
    df=pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["time"]=pd.to_datetime(df["ct"], unit="ms", utc=True)
    df=df.set_index("time").sort_index()
    df=df[~df.index.duplicated(keep="last")]
    return df[["open","high","low","close","volume"]]

def load_1m_resample(symbol, minutes=10, days=7):
    base=load_df(symbol,"1m",days)
    if base.empty: return base
    ohlc={"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out=base.resample(f"{minutes}min", label="right", closed="right").apply(ohlc).dropna()
    return out

def atr(dfi, n=14):
    h,l,c=dfi["high"], dfi["low"], dfi["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    emaf=series.ewm(span=fast, adjust=False).mean()
    emas=series.ewm(span=slow, adjust=False).mean()
    m=emaf-emas; sig=m.ewm(span=signal, adjust=False).mean()
    return m, sig, m-sig

def vwap(dfi):
    tp=(dfi["high"]+dfi["low"]+dfi["close"])/3.0
    cum_pv=(tp*dfi["volume"]).cumsum(); cum_v=dfi["volume"].cumsum().replace(0,np.nan)
    return cum_pv/cum_v

def zigzag(series, atr_series, pct=0.006, atr_mult=1.0):
    idx=series.index
    if len(idx)==0: return []
    swings=[]; direction=None
    extreme_idx=idx[0]; extreme=series.iloc[0]
    def thr(i):
        base=max(pct*series.iloc[i], atr_mult*(atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else 0))
        return base
    for i in range(1,len(idx)):
        p=series.iloc[i]; ts=idx[i]; t=thr(i)
        if direction in (None,"up"):
            if p>extreme: extreme=p; extreme_idx=ts
            elif extreme-p>=t:
                swings.append((extreme_idx,extreme,"H")); direction="down"; extreme_idx=ts; extreme=p
        if direction=="down":
            if p<extreme: extreme=p; extreme_idx=ts
            elif p-extreme>=t:
                swings.append((extreme_idx,extreme,"L")); direction="up"; extreme_idx=ts; extreme=p
    if len(swings)==0 or swings[-1][0]!=extreme_idx:
        swings.append((extreme_idx,extreme,"H" if direction=="down" else "L"))
    return swings

def detect_head_shoulders(df, swings, tol_shoulder=0.25, min_head_gap=0.006, max_neck_slope=0.003, atr_buf_mult=0.25):
    if len(swings)<6: return None
    closes=df["close"]; a=atr(df,14)
    for i in range(0,len(swings)-5):
        window=swings[i:i+6]
        highs=[(t,p) for (t,p,tp) in window if tp=="H"]; lows=[(t,p) for (t,p,tp) in window if tp=="L"]
        if len(highs)>=3 and len(lows)>=2:
            H1t,H1=highs[0]; H2t,H2=highs[1]; H3t,H3=highs[2]; L1t,L1=lows[0]; L2t,L2=lows[1]
            if H2>H1*(1+min_head_gap) and H2>H3*(1+min_head_gap) and abs(H1-H3)/((H1+H3)/2.0)<=tol_shoulder:
                t1,t2=L1t.value,L2t.value
                if t2!=t1:
                    slope=(L2-L1)/(t2-t1)
                    if abs(slope)<=max_neck_slope:
                        nl_now=L1 + slope*(closes.index[-1].value - t1)
                        buf=max(atr_buf_mult*(a.iloc[-1] if not np.isnan(a.iloc[-1]) else 0), 0.0005*closes.iloc[-1])
                        after=df[(df.index>H3t)]
                        brk=after[after["close"] < nl_now - buf]
                        if not brk.empty: return {"pattern":"H&S","direction":"down","neckline":float(nl_now),"break_time":str(brk.index[0])}
        if len(lows)>=3 and len(highs)>=2:
            H1t,H1=highs[0]; H2t,H2=highs[1]; L1t,L1=lows[0]; L2t,L2=lows[1]; L3t,L3=lows[2]
            if L2 < L1*(1- min_head_gap) and L2 < L3*(1- min_head_gap) and abs(L1-L3)/((L1+L3)/2.0)<=tol_shoulder:
                t1,t2=H1t.value,H2t.value
                if t2!=t1:
                    slope=(H2-H1)/(t2-t1)
                    if abs(slope)<=max_neck_slope:
                        nl_now=H1 + slope*(closes.index[-1].value - t1)
                        buf=max(atr_buf_mult*(a.iloc[-1] if not np.isnan(a.iloc[-1]) else 0), 0.0005*closes.iloc[-1])
                        after=df[(df.index>L3t)]
                        brk=after[after["close"] > nl_now + buf]
                        if not brk.empty: return {"pattern":"iH&S","direction":"up","neckline":float(nl_now),"break_time":str(brk.index[0])}
    return None

def detect_double_top_bottom(df, swings, tol=0.004, min_gap=0.006, atr_buf_mult=0.25):
    """Double Top/Bottom with neckline breakout + ATR buffer."""
    if len(swings)<5: return None
    a=atr(df,14); closes=df["close"]
    # DT: H1 ~ H2 within tol, with a low between; breakout below that low
    for i in range(len(swings)-4):
        s=swings[i:i+5]
        highs=[(t,p) for (t,p,tp) in s if tp=="H"]; lows=[(t,p) for (t,p,tp) in s if tp=="L"]
        if len(highs)>=2 and len(lows)>=1:
            (t1,h1),(t2,h2)=highs[0], highs[1]; (tl,nl)=lows[0]
            if abs(h1-h2)/((h1+h2)/2.0) <= tol and h1>nl*(1+min_gap) and h2>nl*(1+min_gap):
                buf=max(atr_buf_mult*(a.iloc[-1] if not np.isnan(a.iloc[-1]) else 0), 0.0005*closes.iloc[-1])
                after=df[(df.index>t2)]
                brk=after[after["close"] < nl - buf]
                if not brk.empty: return {"pattern":"DT","direction":"down","neckline":float(nl),"break_time":str(brk.index[0])}
        # DB: L1 ~ L2 with a high between; breakout above that high
        if len(lows)>=2 and len(highs)>=1:
            (t1,l1),(t2,l2)=lows[0], lows[1]; (th,nh)=highs[0]
            if abs(l1-l2)/((l1+l2)/2.0) <= tol and nh>l1*(1+min_gap) and nh>l2*(1+min_gap):
                buf=max(atr_buf_mult*(a.iloc[-1] if not np.isnan(a.iloc[-1]) else 0), 0.0005*closes.iloc[-1])
                after=df[(df.index>t2)]
                brk=after[after["close"] > nh + buf]
                if not brk.empty: return {"pattern":"DB","direction":"up","neckline":float(nh),"break_time":str(brk.index[0])}
    return None

def htf_snapshot(symbol, interval, days=60):
    if interval=="10m":
        df=load_1m_resample(symbol, minutes=10, days=7)
    else:
        df=load_df(symbol, interval, days)
    if df.empty or len(df)<50: return {"interval":interval, "note":"no data"}
    a=atr(df,14); m,s,h = macd(df["close"])
    vw=vwap(df)
    bias = "UP" if (h.iloc[-1]>0 and df["close"].iloc[-1]>vw.iloc[-1]) else ("DOWN" if (h.iloc[-1]<0 and df["close"].iloc[-1]<vw.iloc[-1]) else "NEUTRAL")
    swings=zigzag(df["close"], a, pct=0.006, atr_mult=1.0)
    pat = detect_head_shoulders(df, swings) or detect_double_top_bottom(df, swings)
    return {"interval": interval, "bias": bias, "macd_hist": float(h.iloc[-1]),
            "vwap": float(vw.iloc[-1]), "price": float(df['close'].iloc[-1]), "pattern": pat}

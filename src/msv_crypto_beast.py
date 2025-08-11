import sys, time, json, math, requests
import pandas as pd, numpy as np
from pathlib import Path

# ====================== CONFIG ======================
SYMBOL = sys.argv[1].upper() if len(sys.argv)>1 else "BTCUSDT"  # e.g., ETHUSDT, SOLUSDT
INTERVAL = "5m"            # Binance klines interval
DAYS = 10                  # lookback days
ORB_MIN = 30               # NY Opening Range minutes
VOL_MULT = 1.5             # vol confirmation vs median(20)
ATR_BUF = 0.25             # BOS buffer = 0.25 * ATR(14), min 5bp
BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000

# Session windows in UTC (crypto 24/7; we still use liquidity windows)
ASIA_START,  ASIA_END   = "00:00", "08:00"
LONDON_START,LONDON_END = "08:00", "13:00"
NY_OPEN_UTC, NY_CLOSE_UTC = "13:30", "20:00"   # 09:30–16:00 ET

# ====================== DATA ======================
def fetch_klines(symbol, interval, start_ms, end_ms):
    out=[]; cur=start_ms
    while cur < end_ms:
        p={"symbol":symbol, "interval":interval, "limit":MAX_LIMIT, "startTime":cur, "endTime":end_ms}
        r=requests.get(BINANCE_URL, params=p, timeout=15); r.raise_for_status()
        batch=r.json()
        if not batch: break
        out.extend(batch)
        last_close=batch[-1][6]
        cur=last_close+1
        if len(batch) < MAX_LIMIT: break
    return out

def load_df(symbol=SYMBOL, interval=INTERVAL, days=DAYS):
    now_ms=int(time.time()*1000); start_ms=now_ms - days*24*60*60*1000
    raw=fetch_klines(symbol, interval, start_ms, now_ms)
    if not raw: raise SystemExit(f"No data from Binance for {symbol}.")
    cols=["open_time","open","high","low","close","volume","close_time","qvol","trades","taker_base","taker_quote","ignore"]
    df=pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df["open_time"]=pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"]=pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df=df.set_index("close_time").sort_index()
    df.index.name="time_utc"
    df=df[~df.index.duplicated(keep="last")].ffill()
    df["day"]=df.index.date
    return df

# ====================== INDICATORS ======================
def atr(dfi, n=14):
    h,l,c = dfi["high"], dfi["low"], dfi["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta=series.diff()
    up=delta.clip(lower=0); down=-delta.clip(upper=0)
    ma_up=up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn=down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn.replace(0,np.nan))
    rsi=100 - (100/(1+rs))
    return rsi.fillna(50)

def stoch_rsi(series, n=14, k=3, d=3):
    r = rsi(series, n)
    ll = r.rolling(n).min(); hh = r.rolling(n).max()
    srsi = (r - ll) / (hh - ll).replace(0,np.nan)
    kline = srsi.rolling(k).mean()
    dline = kline.rolling(d).mean()
    return (srsi.fillna(0.5), kline.fillna(0.5), dline.fillna(0.5))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bbands(series, n=20, mult=2.0):
    mid=series.rolling(n).mean()
    std=series.rolling(n).std(ddof=0)
    upper=mid + mult*std
    lower=mid - mult*std
    width=(upper-lower)/mid
    return mid, upper, lower, width

def anchored_vwap(dfi, anchor_ts):
    sub = dfi[dfi.index >= anchor_ts].copy()
    if sub.empty:
        dfi["avwap"]=np.nan; return dfi["avwap"]
    tp = (sub["high"]+sub["low"]+sub["close"])/3.0
    cum_pv = (tp*sub["volume"]).cumsum()
    cum_v  = sub["volume"].cumsum().replace(0,np.nan)
    vwap = cum_pv / cum_v
    out = pd.Series(index=dfi.index, dtype=float); out.loc[sub.index]=vwap.values
    return out

# ====================== SESSIONS & ORB/PO3/BOS ======================
def between(dfi, day, start_hhmm, end_hhmm):
    s=pd.Timestamp(f"{day} {start_hhmm}", tz="UTC"); e=pd.Timestamp(f"{day} {end_hhmm}", tz="UTC")
    return dfi[(dfi.index>=s)&(dfi.index<e)]

def opening_range(ny_sub, minutes=30):
    if ny_sub.empty: return (np.nan,np.nan,None,None)
    first=ny_sub.index[0]; end=first+pd.Timedelta(minutes=minutes)
    win=ny_sub[(ny_sub.index>=first)&(ny_sub.index<end)]
    if win.empty: return (np.nan,np.nan,None,None)
    return (win["high"].max(), win["low"].min(), first, end)

def po3_for_day(dfi_day, atr_series, day):
    asia=between(dfi_day, day, ASIA_START, ASIA_END)
    if asia.empty: return None
    ah, al = asia["high"].max(), asia["low"].min()
    ny = between(dfi_day, day, NY_OPEN_UTC, NY_CLOSE_UTC)
    _,_,_, orb_end = opening_range(ny, ORB_MIN)
    if orb_end is None: return None
    mid = dfi_day["close"].median()
    buf = max((atr_series.loc[dfi_day.index].median() or 0)*0.25, 0.0005*(mid if pd.notna(mid) else 1))
    w = dfi_day[dfi_day.index <= orb_end]
    sweep_up = w[w["high"] > ah + buf]; sweep_dn = w[w["low"] < al - buf]
    man_dir, man_time = None, None
    if not sweep_up.empty and not sweep_dn.empty:
        man_dir,man_time = ("up",sweep_up.index[0]) if sweep_up.index[0] < sweep_dn.index[0] else ("down",sweep_dn.index[0])
    elif not sweep_up.empty: man_dir,man_time="up",sweep_up.index[0]
    elif not sweep_dn.empty: man_dir,man_time="down",sweep_dn.index[0]
    return {"asia_high": float(ah), "asia_low": float(al), "man_dir": man_dir, "man_time": str(man_time) if man_time else None, "orb_end": str(orb_end)}

def compute_bos_events(ny_df, atr_series, roll=9, atr_buf=ATR_BUF, min_bp=0.0005):
    # swings via causal rolling extrema
    swing_high = ny_df["high"].rolling(roll).max().shift(1)
    swing_low  = ny_df["low"].rolling(roll).min().shift(1)
    events=[]
    for ts,row in ny_df.iterrows():
        atr_val = atr_series.loc[ts] if ts in atr_series.index else np.nan
        buf = max(atr_buf*(atr_val if pd.notna(atr_val) else 0), min_bp*row["close"])
        sh, sl = swing_high.loc[ts], swing_low.loc[ts]
        if pd.notna(sh) and row["close"] > sh + buf:
            events.append({"time":str(ts),"type":"BOS_UP","level":float(sh),"close":float(row["close"])})
        if pd.notna(sl) and row["close"] < sl - buf:
            events.append({"time":str(ts),"type":"BOS_DOWN","level":float(sl),"close":float(row["close"])})
    return events

# ====================== MAIN REPORT ======================
def build_report(symbol=SYMBOL):
    df=load_df(symbol)
    df["atr14"]=atr(df,14)
    days=sorted(df["day"].unique())
    day=days[-1]
    # choose a day with NY bars; fallback to previous
    ny=between(df, day, NY_OPEN_UTC, NY_CLOSE_UTC)
    if ny.empty and len(days)>1:
        day=days[-2]; ny=between(df, day, NY_OPEN_UTC, NY_CLOSE_UTC)
    if ny.empty: raise SystemExit("No NY session slice found in recent data.")
    # ORB
    orb_high,orb_low,orb_start,orb_end = opening_range(ny, ORB_MIN)
    # Anchored VWAP from ORB start
    df["avwap"] = anchored_vwap(df, orb_start) if orb_start else np.nan
    # Indicators on day slice
    day_df = df[df["day"].astype(str).eq(str(day))].copy()
    day_df["rsi14"] = rsi(day_df["close"],14)
    st_srsi, st_k, st_d = stoch_rsi(day_df["close"],14,3,3)
    day_df["srsi"], day_df["k"], day_df["d"] = st_srsi, st_k, st_d
    macd_line, macd_sig, macd_hist = macd(day_df["close"],12,26,9)
    day_df["macd"], day_df["macd_sig"], day_df["macd_hist"] = macd_line, macd_sig, macd_hist
    mid, up, lo, width = bbands(day_df["close"],20,2.0)
    day_df["bb_mid"], day_df["bb_up"], day_df["bb_lo"], day_df["bb_w"] = mid, up, lo, width

    # PO3
    po3 = po3_for_day(day_df, day_df["atr14"], day)

    # BOS after ORB only
    post = ny[ny.index >= orb_end].copy() if orb_end else ny.copy()
    bos_events = compute_bos_events(post, df["atr14"])
    bos_first = next((e for e in bos_events), None)

    # Volume confirmation
    post["med_vol20"]=post["volume"].rolling(20).median()
    last_bar = post.iloc[-1] if not post.empty else day_df.iloc[-1]
    vol_ok_last = (last_bar["volume"] >= VOL_MULT*(last_bar["med_vol20"] if "med_vol20" in last_bar and pd.notna(last_bar["med_vol20"]) else last_bar["volume"]))

    # Confluence scoring (0–100)
    score=0; reasons=[]
    # Trend/BOS bucket (45 pts)
    if bos_first:
        score += 25; reasons.append("BOS detected")
        dir_bos = 1 if bos_first["type"]=="BOS_UP" else -1
        # VWAP agree
        if pd.notna(day_df["avwap"].iloc[-1]):
            if (dir_bos==1 and last_bar["close"]>day_df["avwap"].iloc[-1]) or (dir_bos==-1 and last_bar["close"]<day_df["avwap"].iloc[-1]):
                score += 8; reasons.append("VWAP aligned")
        # MACD agree
        if (dir_bos==1 and day_df["macd_hist"].iloc[-1]>0) or (dir_bos==-1 and day_df["macd_hist"].iloc[-1]<0):
            score += 8; reasons.append("MACD aligned")
        # Volume on last bar
        if vol_ok_last: score += 4; reasons.append("Volume >= 1.5× median")
    # PO3 alignment (15 pts)
    if po3 and po3["man_dir"]:
        # distribution is opposite of manipulation direction
        if bos_first:
            if (po3["man_dir"]=="up" and bos_first["type"]=="BOS_DOWN") or (po3["man_dir"]=="down" and bos_first["type"]=="BOS_UP"):
                score += 12; reasons.append("PO3 aligned with distribution")
            else:
                reasons.append("PO3 misaligned")
        else:
            reasons.append("PO3 present (no BOS)")
    # Bollinger/RSI/StochRSI bucket (25 pts)
    c = last_bar["close"]; bb_up=day_df["bb_up"].iloc[-1]; bb_lo=day_df["bb_lo"].iloc[-1]; rsi14=day_df["rsi14"].iloc[-1]; k=day_df["k"].iloc[-1]; d_=day_df["d"].iloc[-1]
    # Trend-bias from BB midline
    if pd.notna(day_df["bb_mid"].iloc[-1]):
        if c>day_df["bb_mid"].iloc[-1]: score += 5; reasons.append("BB mid above (trend up bias)")
        else: score += 5; reasons.append("BB mid below (trend down bias)")
    # Reversal edge if extreme + cross
    if pd.notna(bb_lo) and c<bb_lo and rsi14<32 and (k>d_) and k<0.25:
        score += 8; reasons.append("Reversal long: band pierce + RSI<32 + K>D from low")
    if pd.notna(bb_up) and c>bb_up and rsi14>68 and (k<d_) and k>0.75:
        score += 8; reasons.append("Reversal short: band pierce + RSI>68 + K<D from high")
    # Momentum confirmation via StochRSI slope
    prev_k=day_df["k"].iloc[-2] if len(day_df)>=2 else k
    if k>prev_k and k>0.55: score += 4; reasons.append("StochRSI rising")
    if k<prev_k and k<0.45: score += 4; reasons.append("StochRSI falling")
    # RSI comfort zone
    if 45<=rsi14<=65: score += 4; reasons.append("RSI in trend zone")

    # ORB retest plan (10 pts)
    entry=None; stop=None; direction=None
    if bos_first:
        if bos_first["type"]=="BOS_UP":
            direction="long"; entry=float(orb_high); stop=float(orb_low); score+=10; reasons.append("Use ORB high/low for R/R")
        else:
            direction="short"; entry=float(orb_low); stop=float(orb_high); score+=10; reasons.append("Use ORB low/high for R/R")
    else:
        # fallback: momentum plan w/ VWAP & MACD
        m_up = (day_df["macd_hist"].iloc[-1]>0) and (c>day_df["avwap"].iloc[-1] if pd.notna(day_df["avwap"].iloc[-1]) else True)
        m_dn = (day_df["macd_hist"].iloc[-1]<0) and (c<day_df["avwap"].iloc[-1] if pd.notna(day_df["avwap"].iloc[-1]) else True)
        if m_up:
            direction="long"; entry=float(day_df["bb_mid"].iloc[-1]); stop=float(day_df["bb_lo"].iloc[-1])
            reasons.append("Momentum long via BB mid/VWAP")
        elif m_dn:
            direction="short"; entry=float(day_df["bb_mid"].iloc[-1]); stop=float(day_df["bb_up"].iloc[-1])
            reasons.append("Momentum short via BB mid/VWAP")

    # Guardrails
    if entry is not None and stop is not None:
        risk=abs(entry-stop)
        tp1 = entry + (risk if direction=="long" else -risk)
        tp2 = entry + (2*risk if direction=="long" else -2*risk)
    else:
        tp1=tp2=None

    # Compose report
    report={
        "symbol":symbol, "pair":"Binance spot", "timeframe":INTERVAL, "tz":"UTC",
        "day":str(day),
        "sessions":{"asia":[ASIA_START,ASIA_END],"london":[LONDON_START,LONDON_END],"ny":[NY_OPEN_UTC,NY_CLOSE_UTC]},
        "orb":{"start":str(orb_start) if orb_start else None, "end":str(orb_end) if orb_end else None,
               "high":float(orb_high) if pd.notna(orb_high) else None, "low":float(orb_low) if pd.notna(orb_low) else None},
        "po3": po3,
        "bos_first": bos_first,
        "indicators":{
            "rsi14": round(float(rsi14),2) if pd.notna(rsi14) else None,
            "stochrsi_k": round(float(k),3) if pd.notna(k) else None,
            "stochrsi_d": round(float(d_),3) if pd.notna(d_) else None,
            "macd_hist": round(float(day_df['macd_hist'].iloc[-1]),4) if pd.notna(day_df['macd_hist'].iloc[-1]) else None,
            "bb_upper": round(float(bb_up),2) if pd.notna(bb_up) else None,
            "bb_lower": round(float(bb_lo),2) if pd.notna(bb_lo) else None,
            "vwap": round(float(day_df["avwap"].iloc[-1]),2) if pd.notna(day_df["avwap"].iloc[-1]) else None,
        },
        "confluence":{"score": int(min(100,max(0,score))), "reasons": reasons[:10]},
        "signal": {
            "direction": direction,
            "entry": round(float(entry),2) if entry else None,
            "stop": round(float(stop),2) if stop else None,
            "tp1": round(float(tp1),2) if tp1 else None,
            "tp2": round(float(tp2),2) if tp2 else None
        }
    }
    return report

if __name__=="__main__":
    rep = build_report(SYMBOL)
    Path("outputs").mkdir(exist_ok=True)
    out = f"outputs/final_trader_report_{SYMBOL}_{rep.get('day','NA')}.json"
    with open(out,"w") as f: json.dump(rep,f,indent=2)
    print("\n=== FINAL TRADER REPORT (CRYPTO BEAST) ===")
    print(f"{rep['symbol']} {rep['day']}  TF {rep['timeframe']}  (UTC sessions)")
    orb=rep["orb"]; print(f"ORB {orb['start']} → {orb['end']}  High={orb['high']}  Low={orb['low']}")
    if rep["po3"] and rep["po3"]["man_dir"]:
        print(f"PO3: Asia {rep['po3']['asia_low']}→{rep['po3']['asia_high']} sweep={rep['po3']['man_dir']} at {rep['po3']['man_time']}")
    print("Indicators:", rep["indicators"])
    print(f"Confluence score: {rep['confluence']['score']}  Reasons: {', '.join(rep['confluence']['reasons'])}")
    s=rep["signal"]
    if s["direction"]:
        print(f"SIGNAL {s['direction'].upper()} | entry {s['entry']} stop {s['stop']} tp1 {s['tp1']} tp2 {s['tp2']}")
    else:
        print("SIGNAL: NONE (insufficient confluence)")
    print(f"Saved: {out}\n")

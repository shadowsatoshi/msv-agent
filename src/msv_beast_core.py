import pandas as pd, numpy as np

# --- indicators ---
def atr(dfi, n=14):
    h,l,c=dfi["high"],dfi["low"],dfi["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n,adjust=False).mean() / dn.ewm(alpha=1/n,adjust=False).mean().replace(0,np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def stoch_rsi(s, n=14, k=3, d=3):
    r=rsi(s,n); ll=r.rolling(n).min(); hh=r.rolling(n).max()
    sr=(r-ll)/(hh-ll).replace(0,np.nan); K=sr.rolling(k).mean(); D=K.rolling(d).mean()
    return sr.fillna(0.5), K.fillna(0.5), D.fillna(0.5)

def macd(s, f=12, sl=26, sig=9):
    ef=s.ewm(span=f,adjust=False).mean(); es=s.ewm(span=sl,adjust=False).mean()
    m=ef-es; sg=m.ewm(span=sig,adjust=False).mean(); return m, sg, m-sg

def bbands(s, n=20, mult=2.0):
    mid=s.rolling(n).mean(); std=s.rolling(n).std(ddof=0)
    up=mid+mult*std; lo=mid-mult*std; w=(up-lo)/mid; return mid,up,lo,w

# --- sessions ---
ASIA_START, ASIA_END = "00:00","08:00"
NY_OPEN_UTC, NY_CLOSE_UTC = "13:30","20:00"

def between(dfi, day, start_hhmm, end_hhmm):
    s=pd.Timestamp(f"{day} {start_hhmm}", tz="UTC"); e=pd.Timestamp(f"{day} {end_hhmm}", tz="UTC")
    return dfi[(dfi.index>=s)&(dfi.index<e)]

def opening_range(ny_df, minutes=30):
    if ny_df.empty: return (np.nan,np.nan,None,None)
    first=ny_df.index[0]; end=first+pd.Timedelta(minutes=minutes)
    win=ny_df[(ny_df.index>=first)&(ny_df.index<end)]
    if win.empty: return (np.nan,np.nan,None,None)
    return (float(win["high"].max()), float(win["low"].min()), first, end)

def po3_for_day(day_df, a14, day, orb_end, atr_buf=0.25):
    asia=between(day_df, day, ASIA_START, ASIA_END)
    if asia.empty or orb_end is None: return None
    ah,al=float(asia["high"].max()), float(asia["low"].min())
    mid=day_df["close"].median()
    buf=max(atr_buf*(a14.loc[day_df.index].median() or 0), 0.0005*(mid if pd.notna(mid) else 1))
    w=day_df[day_df.index<=orb_end]
    up=w[w["high"]>ah+buf]; dn=w[w["low"]<al-buf]
    if not up.empty and not dn.empty:
        return {"asia_high":ah,"asia_low":al,"man_dir":"up" if up.index[0]<dn.index[0] else "down","man_time":str((up.index[0] if up.index[0]<dn.index[0] else dn.index[0])),"orb_end":str(orb_end)}
    if not up.empty: return {"asia_high":ah,"asia_low":al,"man_dir":"up","man_time":str(up.index[0]),"orb_end":str(orb_end)}
    if not dn.empty: return {"asia_high":ah,"asia_low":al,"man_dir":"down","man_time":str(dn.index[0]),"orb_end":str(orb_end)}
    return {"asia_high":ah,"asia_low":al,"man_dir":None,"man_time":None,"orb_end":str(orb_end)}

def compute_bos(ny_df, a14, roll=9, atr_buf=0.25):
    sh=ny_df["high"].rolling(roll).max().shift(1)
    sl=ny_df["low"].rolling(roll).min().shift(1)
    out=[]
    for ts,row in ny_df.iterrows():
        atrv=a14.loc[ts] if ts in a14.index else np.nan
        buf=max(atr_buf*(atrv if pd.notna(atrv) else 0), 0.0005*row["close"])
        if pd.notna(sh.loc[ts]) and row["close"]>sh.loc[ts]+buf:
            out.append({"time":str(ts),"type":"BOS_UP","level":float(sh.loc[ts]),"close":float(row["close"])})
        if pd.notna(sl.loc[ts]) and row["close"]<sl.loc[ts]-buf:
            out.append({"time":str(ts),"type":"BOS_DOWN","level":float(sl.loc[ts]),"close":float(row["close"])})
    return out

def beast_from_df(symbol, df5m, orb_min=30, vol_mult=1.5, atr_buf=0.25):
    if df5m.empty or len(df5m)<60: 
        return {"symbol":symbol,"note":"not enough bars"}
    df=df5m.copy()
    df["day"]=df.index.date
    a14=atr(df,14)
    day=sorted(df["day"].unique())[-1]
    ny=between(df, day, NY_OPEN_UTC, NY_CLOSE_UTC)
    if ny.empty and len(df["day"].unique())>1:
        day=sorted(df["day"].unique())[-2]; ny=between(df, day, NY_OPEN_UTC, NY_CLOSE_UTC)
    orb_hi,orb_lo,orb_start,orb_end = opening_range(ny, orb_min)
    # indicators
    rsi14 = rsi(df["close"],14); srsi,k,d = stoch_rsi(df["close"],14,3,3)
    m,sg,h = macd(df["close"],12,26,9)
    mid,up,lo,w = bbands(df["close"],20,2.0)
    df["rsi14"],df["k"],df["d"],df["macd_hist"],df["bb_mid"],df["bb_up"],df["bb_lo"]=rsi14,k,d,h,mid,up,lo
    # PO3
    po3 = po3_for_day(df[df["day"].astype(str).eq(str(day))], a14, day, orb_end, atr_buf)
    # BOS after ORB
    post = ny[ny.index>=orb_end].copy() if orb_end else ny.copy()
    post["med_vol20"]=post["volume"].rolling(20).median()
    bos = compute_bos(post, a14, 9, atr_buf)
    first = next(iter(bos), None)
    # volume confirm on last bar
    last = post.iloc[-1] if not post.empty else df.iloc[-1]
    vol_ok = True
    if "med_vol20" in last and pd.notna(last["med_vol20"]):
        vol_ok = last["volume"] >= vol_mult*last["med_vol20"]
    # confluence
    score=0; reasons=[]
    if first:
        score+=25; reasons.append("BOS detected")
        dirn = 1 if first["type"]=="BOS_UP" else -1
        if (dirn==1 and df["close"].iloc[-1]>df["bb_mid"].iloc[-1]) or (dirn==-1 and df["close"].iloc[-1]<df["bb_mid"].iloc[-1]): score+=8; reasons.append("BB mid aligned")
        if (dirn==1 and df["macd_hist"].iloc[-1]>0) or (dirn==-1 and df["macd_hist"].iloc[-1]<0): score+=8; reasons.append("MACD aligned")
        if vol_ok: score+=4; reasons.append("Volume ok")
    if po3 and po3.get("man_dir") and first:
        if (po3["man_dir"]=="up" and first["type"]=="BOS_DOWN") or (po3["man_dir"]=="down" and first["type"]=="BOS_UP"):
            score+=12; reasons.append("PO3 aligned")
    c=df["close"].iloc[-1]; bb_up=df["bb_up"].iloc[-1]; bb_lo=df["bb_lo"].iloc[-1]; r=df["rsi14"].iloc[-1]
    if c>df["bb_mid"].iloc[-1]: score+=5; reasons.append("Trend up bias") 
    else: score+=5; reasons.append("Trend down bias")
    if pd.notna(bb_lo) and c<bb_lo and r<32: score+=8; reasons.append("Reversal long setup")
    if pd.notna(bb_up) and c>bb_up and r>68: score+=8; reasons.append("Reversal short setup")
    prevk=df["k"].iloc[-2] if len(df)>=2 else df["k"].iloc[-1]
    if df["k"].iloc[-1]>prevk and df["k"].iloc[-1]>0.55: score+=4; reasons.append("StochRSI rising")
    if df["k"].iloc[-1]<prevk and df["k"].iloc[-1]<0.45: score+=4; reasons.append("StochRSI falling")
    if 45<=r<=65: score+=4; reasons.append("RSI in trend zone")
    # plan
    entry=stop=tp1=tp2=None; direction=None
    if first:
        if first["type"]=="BOS_UP":
            direction="long"; entry=orb_hi; stop=orb_lo
        else:
            direction="short"; entry=orb_lo; stop=orb_hi
        R=abs(entry-stop); tp1=entry+(R if direction=="long" else -R); tp2=entry+(2*R if direction=="long" else -2*R)
        score+=10; reasons.append("ORB R/R plan")
    report={"symbol":symbol,"timeframe":"5m","day":str(day),
            "orb":{"start":str(orb_start),"end":str(orb_end),"high":orb_hi,"low":orb_lo},
            "po3":po3,
            "confluence":{"score":int(min(100,max(0,score))),"reasons":reasons[:10]},
            "signal":{"direction":direction,"entry":round(entry,2) if entry else None,"stop":round(stop,2) if stop else None,
                      "tp1":round(tp1,2) if tp1 else None,"tp2":round(tp2,2) if tp2 else None}}
    return report

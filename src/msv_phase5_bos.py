import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, json, yfinance as yf, os

# ---------- CONFIG ----------
SYMBOL="ES=F"                # switch to "SPY" if ES=F misbehaves
INTERVAL="5m"; PERIOD="14d"
ASIA_START,ASIA_END="00:00","08:00"
LON_START,LON_END="08:00","13:00"
NY_OPEN_UTC,NY_CLOSE_UTC="13:30","20:00"
ORB_MIN=30

# ---------- DATA ----------
df=yf.download(SYMBOL,interval=INTERVAL,period=PERIOD,prepost=True,auto_adjust=False)
if df.empty: raise SystemExit("No data. Try SPY.")
df.columns=[str(c[0]).lower() for c in df.columns] if isinstance(df.columns,pd.MultiIndex) else [str(c).lower() for c in df.columns]
df.index = (df.index.tz_localize("America/New_York") if df.index.tz is None else df.index).tz_convert("UTC")
df=df[~df.index.duplicated(keep="last")].sort_index().ffill(); df["day"]=df.index.date

# indicators
def atr(dfi,n=14):
    h,l,c=dfi["high"],dfi["low"],dfi["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()
df["atr14"]=atr(df,14)

# sessions
def sess(ts):
    t=ts.strftime("%H:%M")
    if ASIA_START<=t<ASIA_END: return "ASIA"
    if LON_START<=t<LON_END: return "LONDON"
    if NY_OPEN_UTC<=t<NY_CLOSE_UTC: return "NY"
    return "OFF"
df["session"]=[sess(ts) for ts in df.index]

# daily levels
daily=df.groupby("day").agg(day_high=("high","max"),day_low=("low","min"))
df=df.join(daily,on="day")
pdhpdl=daily.shift(1).rename(columns={"day_high":"pdh","day_low":"pdl"})
df=df.join(pdhpdl,on="day")
asia=df[df.session.eq("ASIA")].groupby("day").agg(asia_high=("high","max"),asia_low=("low","min"))
df=df.join(asia,on="day")

# ORB
def ny_slice(d):
    s=pd.Timestamp(f"{d} {NY_OPEN_UTC}",tz="UTC"); e=pd.Timestamp(f"{d} {NY_CLOSE_UTC}",tz="UTC")
    return df[(df.index>=s)&(df.index<e)]
def opening_range(sub,minutes=30):
    if sub.empty: return (np.nan,np.nan,None,None)
    first=sub.index[0]; end=first+pd.Timedelta(minutes=minutes)
    win=sub[(sub.index>=first)&(sub.index<end)]
    if win.empty: return (np.nan,np.nan,None,None)
    return (win["high"].max(),win["low"].min(),first,end)
recs=[]
for d in sorted(df["day"].unique()):
    ny=ny_slice(d); hi,lo,st,en=opening_range(ny,ORB_MIN); recs.append((d,hi,lo,st,en))
orb=pd.DataFrame(recs,columns=["day","orb_high","orb_low","orb_start","orb_end"]).set_index("day")
df=df.join(orb,on="day")

# ---------- PHASE 5: TRUE BOS ----------
# pivots via centered rolling max/min
def add_pivots(dfi, left=3, right=3):
    win=left+right+1
    rh=dfi["high"].rolling(win,center=True).max()
    rl=dfi["low"].rolling(win,center=True).min()
    dfi["pivot_high"]=(dfi["high"]==rh)
    dfi["pivot_low"] =(dfi["low"] ==rl)
    return dfi

# choose latest day with ORB
cands=[d for d in df["day"].unique() if df.loc[df["day"].eq(d),"orb_start"].notna().any()]
if not cands: raise SystemExit("No NY ORB found.")
day=cands[-1]
view=df[df["day"].eq(day)].copy()
view=add_pivots(view,3,3)

# walk forward, track last swing highs/lows; flag BOS when close breaks prior swing with buffer
events=[]; last_swing_high=None; last_swing_low=None
for ts,row in view.iterrows():
    if row["pivot_high"]: last_swing_high=float(row["high"])
    if row["pivot_low"] : last_swing_low =float(row["low"])
    if pd.notna(row["atr14"]):
        buf=max(0.25*row["atr14"], 0.0005*row["close"])  # ATR buffer, min 5bp
        # BOS up: close > last_swing_high + buffer
        if last_swing_high and row["close"]>last_swing_high+buf:
            events.append({"time":str(ts),"type":"BOS_UP","level":last_swing_high,"close":float(row["close"])})
            last_swing_low=None   # reset opposite swing
        # BOS down: close < last_swing_low - buffer
        if last_swing_low and row["close"]<last_swing_low-buf:
            events.append({"time":str(ts),"type":"BOS_DOWN","level":last_swing_low,"close":float(row["close"])})
            last_swing_high=None  # reset opposite swing

# ---------- OUTPUTS ----------
os.makedirs("outputs",exist_ok=True)
# plot day with BOS marks
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(view.index,view["close"])
# shade sessions
def span(h1,h2): ax.axvspan(pd.Timestamp(f"{day} {h1}",tz="UTC"), pd.Timestamp(f"{day} {h2}",tz="UTC"), alpha=0.08)
span(ASIA_START,ASIA_END); span(LON_START,LON_END); span(NY_OPEN_UTC,NY_CLOSE_UTC)
# levels
for lvl in ["asia_low","asia_high","pdl","pdh","orb_low","orb_high"]:
    v=view[lvl].iloc[0]
    if pd.notna(v): ax.axhline(v)
# BOS markers
for e in events:
    x=pd.Timestamp(e["time"])
    ax.axvline(x, alpha=0.2)
    ax.text(x, view["close"].min(), e["type"], rotation=90, va="bottom")
ax.set_title(f"{SYMBOL} — {day} (UTC): TRUE BOS events + Sessions & Levels")
ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Price")
plt.tight_layout(); plt.savefig("outputs/day_bos.png", dpi=150); plt.close()

# save JSON
with open("outputs/bos_events.json","w") as f:
    json.dump({"symbol":SYMBOL,"day":str(day),"events":events}, f, indent=2)

print(f"Saved: outputs/day_bos.png  and  outputs/bos_events.json")
print(f"BOS events: {len(events)}")
if events[:3]: 
    print("First events:", events[:3])

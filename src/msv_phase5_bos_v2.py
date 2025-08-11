import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, json, yfinance as yf, os

# ---------------- CONFIG ----------------
SYMBOL="ES=F"                 # use "SPY" if ES=F is quirky
INTERVAL="5m"; PERIOD="14d"
ASIA_START,ASIA_END="00:00","08:00"
LON_START,LON_END="08:00","13:00"
NY_OPEN_UTC,NY_CLOSE_UTC="13:30","20:00"
ORB_MIN=30
ZIGZAG_PCT=0.006              # 0.6% move threshold
ZIGZAG_ATR_MULT=1.0           # or 1× ATR14, whichever is larger
BOS_BUF_ATR=0.25              # close must exceed swing by 0.25×ATR
MIN_BARS_BETWEEN=6            # cool-down between BOS marks
MAX_LABELS=6                  # keep the chart readable

# ---------------- DATA ----------------
df=yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, prepost=True, auto_adjust=False)
if df.empty: raise SystemExit("No data. Try SPY.")
df.columns=[str(c[0]).lower() for c in df.columns] if isinstance(df.columns,pd.MultiIndex) else [str(c).lower() for c in df.columns]
df.index=(df.index.tz_localize("America/New_York") if df.index.tz is None else df.index).tz_convert("UTC")
df=df[~df.index.duplicated(keep="last")].sort_index().ffill()
df["day"]=df.index.date

def atr(dfi,n=14):
    h,l,c=dfi["high"],dfi["low"],dfi["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()
df["atr14"]=atr(df,14)

def sess(ts):
    t=ts.strftime("%H:%M")
    if ASIA_START<=t<ASIA_END: return "ASIA"
    if LON_START<=t<LON_END: return "LONDON"
    if NY_OPEN_UTC<=t<NY_CLOSE_UTC: return "NY"
    return "OFF"
df["session"]=[sess(ts) for ts in df.index]

# prior day & Asia levels
daily=df.groupby("day").agg(day_high=("high","max"), day_low=("low","min"))
df=df.join(daily,on="day")
pdhpdl=daily.shift(1).rename(columns={"day_high":"pdh","day_low":"pdl"})
df=df.join(pdhpdl,on="day")
asia=df[df.session.eq("ASIA")].groupby("day").agg(asia_high=("high","max"), asia_low=("low","min"))
df=df.join(asia,on="day")

# ORB for NY
def ny_slice(d):
    s=pd.Timestamp(f"{d} {NY_OPEN_UTC}",tz="UTC"); e=pd.Timestamp(f"{d} {NY_CLOSE_UTC}",tz="UTC")
    return df[(df.index>=s)&(df.index<e)]
def opening_range(sub, minutes=30):
    if sub.empty: return (np.nan,np.nan,None,None)
    first=sub.index[0]; end=first+pd.Timedelta(minutes=minutes)
    win=sub[(sub.index>=first)&(sub.index<end)]
    if win.empty: return (np.nan,np.nan,None,None)
    return (win["high"].max(), win["low"].min(), first, end)
recs=[]
for d in sorted(df["day"].unique()):
    ny=ny_slice(d); hi,lo,st,en=opening_range(ny, ORB_MIN); recs.append((d,hi,lo,st,en))
orb=pd.DataFrame(recs, columns=["day","orb_high","orb_low","orb_start","orb_end"]).set_index("day")
df=df.join(orb,on="day")

# ---------------- ZigZag swings (ATR- and %-aware) ----------------
def zigzag(close, atr, pct=ZIGZAG_PCT, atr_mult=ZIGZAG_ATR_MULT):
    idx=close.index
    if len(idx)==0: return []
    swings=[]  # list of (ts, price, type) where type in {"H","L"}
    last_idx=idx[0]; last_price=close.iloc[0]; direction=None  # None, "up", "down"
    extreme_idx=last_idx; extreme_price=last_price

    def threshold(i):
        base=max(pct*close.iloc[i], atr_mult*(atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0))
        return base

    for i in range(1, len(idx)):
        p=close.iloc[i]; ts=idx[i]; th=threshold(i)
        if direction in (None,"up"):
            # looking for higher highs; flip if drawdown exceeds th
            if p>extreme_price:
                extreme_price=p; extreme_idx=ts
            elif extreme_price - p >= th:
                # confirmed swing high
                swings.append((extreme_idx, extreme_price, "H"))
                direction="down"
                extreme_idx=ts; extreme_price=p
        if direction=="down":
            if p<extreme_price:
                extreme_price=p; extreme_idx=ts
            elif p - extreme_price >= th:
                # confirmed swing low
                swings.append((extreme_idx, extreme_price, "L"))
                direction="up"
                extreme_idx=ts; extreme_price=p
    # ensure last extreme recorded
    if len(swings)==0 or swings[-1][0]!=extreme_idx:
        swings.append((extreme_idx, extreme_price, "H" if direction=="down" else "L"))
    return swings

# choose latest complete day with ORB
cands=[d for d in df["day"].unique() if df.loc[df["day"].eq(d),"orb_end"].notna().any()]
if not cands: raise SystemExit("No NY ORB found.")
day=cands[-1]
day_df=df[df["day"].eq(day)].copy()
ny=ny_slice(day)
swings=zigzag(day_df["close"], day_df["atr14"])

# Map last opposite swing during NY session and detect BOS with buffer
events=[]; last_high=None; last_low=None; bars_since=MIN_BARS_BETWEEN+1
for i, (ts, price, typ) in enumerate(swings):
    # update latest swings up to current ts
    if typ=="H": last_high=price
    if typ=="L": last_low=price

# iterate bars only in NY session, after ORB start
start=day_df["orb_start"].iloc[0]; end=day_df["orb_end"].iloc[0]
for i, (ts, row) in enumerate(ny.iterrows()):
    bars_since+=1
    atr_val=row["atr14"] if not np.isnan(row["atr14"]) else 0.0
    buf=max(BOS_BUF_ATR*atr_val, 0.0005*row["close"])  # 0.25×ATR or 5bp min

    # update swings as time advances
    swings_up_to=[s for s in swings if s[0] <= ts]
    if swings_up_to:
        # last swing highs/lows up to now
        highs=[p for (t,p,ty) in swings_up_to if ty=="H"]
        lows =[p for (t,p,ty) in swings_up_to if ty=="L"]
        last_high = highs[-1] if highs else last_high
        last_low  = lows[-1]  if lows  else last_low

    # require a recent reference
    if last_high and row["close"]>last_high+buf and bars_since>=MIN_BARS_BETWEEN:
        events.append({"time":str(ts),"type":"BOS_UP","level":float(last_high),"close":float(row["close"])})
        bars_since=0
    elif last_low and row["close"]<last_low-buf and bars_since>=MIN_BARS_BETWEEN:
        events.append({"time":str(ts),"type":"BOS_DOWN","level":float(last_low),"close":float(row["close"])})
        bars_since=0

# keep it readable
events=events[:MAX_LABELS]

# ---------------- OUTPUTS ----------------
os.makedirs("outputs",exist_ok=True)
# Plot
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(day_df.index, day_df["close"])
def span(h1,h2): ax.axvspan(pd.Timestamp(f"{day} {h1}",tz="UTC"), pd.Timestamp(f"{day} {h2}",tz="UTC"), alpha=0.08)
span(ASIA_START,ASIA_END); span(LON_START,LON_END); span(NY_OPEN_UTC,NY_CLOSE_UTC)
for lvl in ["asia_low","asia_high","pdl","pdh","orb_low","orb_high"]:
    v=day_df[lvl].iloc[0]
    if pd.notna(v): ax.axhline(v)
# BOS markers
for e in events:
    x=pd.Timestamp(e["time"])
    ax.axvline(x, alpha=0.25)
    ax.text(x, day_df["close"].min(), e["type"], rotation=90, va="bottom")
ax.set_title(f"{SYMBOL} — {day} (UTC): Clean TRUE BOS (ZigZag, ATR buffer, NY only)")
ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Price")
plt.tight_layout(); plt.savefig("outputs/day_bos_clean.png", dpi=150); plt.close()

# JSON
with open("outputs/bos_events_clean.json","w") as f:
    json.dump({"symbol":SYMBOL,"day":str(day),"events":events}, f, indent=2)
print("Saved: outputs/day_bos_clean.png  and  outputs/bos_events_clean.json")
print("Events:", events)

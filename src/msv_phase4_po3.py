import matplotlib
matplotlib.use("Agg")  # headless plotting (no window)
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import yfinance as yf
import os

# ------------ CONFIG ------------
SYMBOL = "ES=F"   # switch to "SPY" if needed
INTERVAL = "5m"; PERIOD = "14d"; ORB_MIN = 30
ASIA_START, ASIA_END = "00:00", "08:00"
LON_START,  LON_END  = "08:00", "13:00"
NY_OPEN_UTC, NY_CLOSE_UTC = "13:30", "20:00"

# ------------ DOWNLOAD & CLEAN ------------
df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, prepost=True, auto_adjust=False)
if df.empty: raise SystemExit("No data. Try SYMBOL='SPY' or PERIOD='7d'.")
if isinstance(df.columns, pd.MultiIndex): df.columns = [str(c[0]).lower() for c in df.columns]
else: df.columns = [str(c).lower() for c in df.columns]
if df.index.tz is None: df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
else: df.index = df.index.tz_convert("UTC")
df = df[~df.index.duplicated(keep="last")].sort_index().ffill()
df["day"] = df.index.date

# ATR for small buffers
def atr(dfi, n=14):
    h,l,c = dfi["high"], dfi["low"], dfi["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()
df["atr14"] = atr(df,14)

# ------------ Sessions & Ranges ------------
def sess(ts):
    t = ts.strftime("%H:%M")
    if ASIA_START<=t<ASIA_END: return "ASIA"
    if LON_START <=t<LON_END:  return "LONDON"
    if NY_OPEN_UTC<=t<NY_CLOSE_UTC: return "NY"
    return "OFF"
df["session"] = [sess(ts) for ts in df.index]

daily = df.groupby("day").agg(day_high=("high","max"), day_low=("low","min"))
df = df.join(daily, on="day")
pdhpdl = daily.shift(1).rename(columns={"day_high":"pdh","day_low":"pdl"})
df = df.join(pdhpdl, on="day")
asia = df[df.session.eq("ASIA")].groupby("day").agg(asia_high=("high","max"), asia_low=("low","min"))
df = df.join(asia, on="day")

# ORB per day
def ny_slice(d):
    s = pd.Timestamp(f"{d} {NY_OPEN_UTC}", tz="UTC")
    e = pd.Timestamp(f"{d} {NY_CLOSE_UTC}", tz="UTC")
    return df[(df.index>=s)&(df.index<e)]
def opening_range(sub, minutes=30):
    if sub.empty: return (np.nan, np.nan, None, None)
    first=sub.index[0]; end=first+pd.Timedelta(minutes=minutes)
    win=sub[(sub.index>=first)&(sub.index<end)]
    if win.empty: return (np.nan, np.nan, None, None)
    return (win["high"].max(), win["low"].min(), first, end)

recs=[]
for d in sorted(df["day"].unique()):
    ny = ny_slice(d)
    hi,lo,st,en = opening_range(ny, ORB_MIN)
    recs.append((d,hi,lo,st,en))
orb = pd.DataFrame(recs, columns=["day","orb_high","orb_low","orb_start","orb_end"]).set_index("day")
df = df.join(orb, on="day")

# ------------ PO3 tagging ------------
def po3_for_day(d):
    day_rows = df[df["day"].eq(d)]
    if day_rows.empty or pd.isna(day_rows["asia_high"]).all() or pd.isna(day_rows["orb_end"]).all():
        return None
    acc_start = pd.Timestamp(f"{d} {ASIA_START}", tz="UTC")
    acc_end   = pd.Timestamp(f"{d} {ASIA_END}",   tz="UTC")

    # buffer for sweep detection
    mid_close = day_rows["close"].median()
    buf = (day_rows["atr14"].median() or 0) * 0.25
    buf = max(buf, 0.0005 * (mid_close if pd.notna(mid_close) else 1))  # min 5bp

    ah = day_rows["asia_high"].iloc[0]
    al = day_rows["asia_low"].iloc[0]
    orb_end = day_rows["orb_end"].iloc[0]

    man_dir, man_start = None, None
    if pd.notna(ah) and pd.notna(al) and pd.notna(orb_end):
        w = day_rows[day_rows.index <= orb_end]
        sweep_up = w[w["high"] > ah + buf]
        sweep_dn = w[w["low"]  < al - buf]
        if not sweep_up.empty and not sweep_dn.empty:
            man_dir, man_start = ("up", sweep_up.index[0]) if sweep_up.index[0] < sweep_dn.index[0] else ("down", sweep_dn.index[0])
        elif not sweep_up.empty:
            man_dir, man_start = "up", sweep_up.index[0]
        elif not sweep_dn.empty:
            man_dir, man_start = "down", sweep_dn.index[0]

    dist_start = None
    if man_dir and pd.notna(orb_end):
        ny = ny_slice(d)
        post = ny[ny.index >= orb_end]
        if man_dir == "up":
            # opposite: break ORB low
            dist_trig = post[post["close"] < day_rows["orb_low"].iloc[0]]
        else:
            # opposite: break ORB high
            dist_trig = post[post["close"] > day_rows["orb_high"].iloc[0]]
        if not dist_trig.empty:
            dist_start = dist_trig.index[0]

    return {
        "acc":  (acc_start, acc_end),
        "man":  (man_start, orb_end) if man_start is not None else None,
        "dist": (dist_start, pd.Timestamp(f"{d} {NY_CLOSE_UTC}", tz="UTC")) if dist_start is not None else None,
        "man_dir": man_dir
    }

# choose a day with ORB present
cands=[d for d in df["day"].unique() if df.loc[df["day"].eq(d),"orb_start"].notna().any()]
if not cands: raise SystemExit("No NY ORB found. Try SPY.")
sample_day = cands[-1]
tags = po3_for_day(sample_day)

print(f"PO3 for {sample_day}:")
if tags:
    print("  ACC:", tags["acc"])
    print("  MAN:", tags["man"], "dir:", tags["man_dir"])
    print("  DIST:", tags["dist"])
else:
    print("  (no tags)")

# ------------ PLOT (saved only) ------------
os.makedirs("outputs", exist_ok=True)
view = df[df["day"].eq(sample_day)]
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(view.index, view["close"])

def shade(start, end, alpha=0.08):
    if start and end: ax.axvspan(start, end, alpha=alpha)
def ts(day, hhmm): return pd.Timestamp(f"{day} {hhmm}", tz="UTC")

shade(ts(sample_day, ASIA_START), ts(sample_day, ASIA_END))
shade(ts(sample_day, LON_START),  ts(sample_day, LON_END))
shade(ts(sample_day, NY_OPEN_UTC), ts(sample_day, NY_CLOSE_UTC))

for lvl in ["asia_low","asia_high","pdl","pdh","orb_low","orb_high"]:
    v = view[lvl].iloc[0]
    if pd.notna(v): ax.axhline(v)

if tags:
    acc, man, dist = tags["acc"], tags["man"], tags["dist"]
    if acc:  ax.axvspan(acc[0], acc[1], alpha=0.10)
    if man:  ax.axvspan(man[0], man[1], alpha=0.15)
    if dist: ax.axvspan(dist[0], dist[1], alpha=0.10)
    if acc:  ax.text(acc[0],  view["close"].min(), "ACC",  va="bottom")
    if man:  ax.text(man[0],  view["close"].min(), "MAN",  va="bottom")
    if dist: ax.text(dist[0], view["close"].min(), "DIST", va="bottom")

ax.set_title(f"{SYMBOL} — {sample_day} (UTC): PO3 (ACC/MAN/DIST), Sessions, Asia, PDH/PDL, ORB")
ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Price")
plt.tight_layout()
plt.savefig("outputs/day_po3.png", dpi=150)
plt.close()

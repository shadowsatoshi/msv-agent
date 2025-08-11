import pandas as pd, numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
SYMBOL = "ES=F"         # if this fails, change to "SPY"
INTERVAL = "5m"
PERIOD = "14d"
ORB_MIN = 30

# Sessions in UTC
ASIA_START, ASIA_END       = "00:00", "08:00"
LONDON_START, LONDON_END   = "08:00", "13:00"
NY_OPEN_UTC, NY_CLOSE_UTC  = "13:30", "20:00"   # 09:30–16:00 ET in UTC

# ---------- DOWNLOAD & NORMALIZE (UTC) ----------
df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, prepost=True, auto_adjust=False)
if df.empty:
    raise SystemExit("No data. Try SYMBOL='SPY' or reduce PERIOD (e.g., '7d').")

# Handle yfinance MultiIndex columns gracefully
if isinstance(df.columns, pd.MultiIndex):
    # typically [('Open','ES=F'), ('High','ES=F'), ...]
    df.columns = [str(levels[0]).lower() for levels in df.columns]
else:
    df.columns = [str(c).lower() for c in df.columns]

# Ensure UTC index
if df.index.tz is None:
    df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
else:
    df.index = df.index.tz_convert("UTC")

df = df[~df.index.duplicated(keep="last")].sort_index().ffill()
df["day"] = df.index.date

# ---------- SESSIONS & RANGES ----------
def label_session(ts):
    t = ts.strftime("%H:%M")
    if ASIA_START   <= t < ASIA_END:     return "ASIA"
    if LONDON_START <= t < LONDON_END:   return "LONDON"
    if NY_OPEN_UTC  <= t < NY_CLOSE_UTC: return "NY"
    return "OFF"

df["session"] = [label_session(ts) for ts in df.index]

# prior-day high/low (UTC calendar day)
daily = df.groupby("day").agg(day_high=("high","max"), day_low=("low","min"))
df = df.join(daily, on="day")
pdhpdl = daily.shift(1).rename(columns={"day_high":"pdh","day_low":"pdl"})
df = df.join(pdhpdl, on="day")

# Asia range per day
asia = df[df.session.eq("ASIA")].groupby("day").agg(
    asia_high=("high","max"),
    asia_low=("low","min")
)
df = df.join(asia, on="day")

# ---------- NY ORB ----------
def ny_slice(d):
    s = pd.Timestamp(f"{d} {NY_OPEN_UTC}", tz="UTC")
    e = pd.Timestamp(f"{d} {NY_CLOSE_UTC}", tz="UTC")
    return df[(df.index>=s) & (df.index<e)]

def opening_range(sub, minutes=30):
    if sub.empty: return (np.nan, np.nan, None, None)
    first = sub.index[0]
    end = first + pd.Timedelta(minutes=minutes)
    win = sub[(sub.index>=first) & (sub.index<end)]
    if win.empty: return (np.nan, np.nan, None, None)
    return (win["high"].max(), win["low"].min(), first, end)

records=[]
for d in sorted(df["day"].unique()):
    ny = ny_slice(d)
    hi, lo, st, en = opening_range(ny, ORB_MIN)
    records.append((d, hi, lo, st, en))
orb = pd.DataFrame(records, columns=["day","orb_high","orb_low","orb_start","orb_end"]).set_index("day")
df = df.join(orb, on="day")

# pick a sample day with ORB present
candidates = [d for d in df["day"].unique() if df.loc[df["day"].eq(d),"orb_start"].notna().any()]
if not candidates:
    raise SystemExit("No NY ORB found. Switch SYMBOL to 'SPY' and rerun.")
sample_day = candidates[-1]
view = df[df["day"].eq(sample_day)].copy()

print("Sample day:", sample_day)
print("Asia range:", view["asia_low"].iloc[0], "→", view["asia_high"].iloc[0])
print("Prior day H/L:", view["pdl"].iloc[0], "→", view["pdh"].iloc[0])
print("NY ORB:", view["orb_low"].iloc[0], "→", view["orb_high"].iloc[0])
print("✅ Phase 0–3 ready. A chart will open and a PNG will be saved.")

# ---------- PLOT ----------
os.makedirs("outputs", exist_ok=True); os.makedirs("data", exist_ok=True)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(view.index, view["close"])

# shade sessions (Asia, London, NY)
def shade(d, start, end):
    s = pd.Timestamp(f"{d} {start}", tz="UTC")
    e = pd.Timestamp(f"{d} {end}", tz="UTC")
    ax.axvspan(s, e, alpha=0.08)

shade(sample_day, ASIA_START, ASIA_END)
shade(sample_day, LONDON_START, LONDON_END)
shade(sample_day, NY_OPEN_UTC, NY_CLOSE_UTC)

# lines: Asia range, PDH/PDL, ORB
for lvl in ["asia_low","asia_high","pdl","pdh","orb_low","orb_high"]:
    val = view[lvl].iloc[0]
    if pd.notna(val): ax.axhline(val)

# highlight the opening 30 minutes
if pd.notna(view["orb_start"].iloc[0]) and pd.notna(view["orb_end"].iloc[0]):
    ax.axvspan(view["orb_start"].iloc[0], view["orb_end"].iloc[0], alpha=0.10)

ax.set_title(f"{SYMBOL} — {sample_day} (UTC): Sessions + Asia range + PDH/PDL + NY ORB")
ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Price")
plt.tight_layout()
plt.savefig("outputs/day_overview.png", dpi=150)
df.to_csv("data/phase0_3_snapshot.csv")
plt.show()

import pandas as pd, numpy as np, yfinance as yf, json
from pathlib import Path

SYMBOL="SPY"                 # simple + liquid; switch later if you want
INTERVAL="5m"; PERIOD="10d"
NY_TZ="America/New_York"
OPEN="09:30"; CLOSE="16:00"; ORB_MIN=30
VOL_MULT=1.5                 # volume confirm vs 20-bar median
ATR_BUF=0.25                 # 0.25×ATR buffer for BOS

# -------- data --------
df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD, auto_adjust=False, prepost=False)
if df.empty:
    raise SystemExit("No data.")

# Handle MultiIndex columns cleanly
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).title() for c in df.columns]  # Open, High, Low, Close, Adj Close, Volume

# Convert to Eastern Time for readability
df.index = (df.index.tz_localize(NY_TZ) if df.index.tz is None else df.index.tz_convert(NY_TZ))
df = df[~df.index.duplicated(keep="last")].sort_index()

# pick latest regular-hours day
day = df.index[-1].date()
day_df = df[df.index.date==day].between_time(OPEN, CLOSE)
if day_df.empty:
    day = (df.index[-1] - pd.Timedelta(days=1)).date()
    day_df = df[df.index.date==day].between_time(OPEN, CLOSE)
    if day_df.empty:
        raise SystemExit("No regular NY session found in window.")

# ATR(14) on 5m
h,l,c = day_df["High"], day_df["Low"], day_df["Close"]; pc = c.shift(1)
tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
atr = tr.ewm(span=14, adjust=False).mean()

# Opening range (first ORB_MIN minutes)
orb_start = day_df.index[0]
orb_end   = orb_start + pd.Timedelta(minutes=ORB_MIN)
win = day_df[(day_df.index>=orb_start) & (day_df.index<orb_end)]
orb_high, orb_low = float(win["High"].max()), float(win["Low"].min())

# simple swings via rolling extrema (causal)
roll = 9
swing_high = day_df["High"].rolling(roll).max().shift(1)
swing_low  = day_df["Low"].rolling(roll).min().shift(1)

# After ORB → first BOS with ATR buffer + volume confirm
post = day_df[day_df.index>=orb_end].copy()
median_vol = post["Volume"].rolling(20).median()

signal = None
for ts, row in post.iterrows():
    buf = max(ATR_BUF*atr.loc[ts], 0.0005*row["Close"])  # min 5bp
    sh, sl = swing_high.loc[ts], swing_low.loc[ts]
    mv = median_vol.loc[ts] if pd.notna(median_vol.loc[ts]) else row["Volume"]
    vol_ok = row["Volume"] >= VOL_MULT * mv

    if pd.notna(sh) and row["Close"] > sh + buf and vol_ok:
        # BOS_UP → long on pullback to ORB high with SL below ORB low
        entry, stop = orb_high, orb_low
        risk = entry - stop
        signal = {"direction":"long","reason":"BOS up + volume",
                  "entry":round(entry,2),"stop":round(stop,2),
                  "tp1":round(entry+risk,2),"tp2":round(entry+2*risk,2),
                  "bos_time":str(ts)}
        break

    if pd.notna(sl) and row["Close"] < sl - buf and vol_ok:
        # BOS_DOWN → short on pullback to ORB low with SL above ORB high
        entry, stop = orb_low, orb_high
        risk = stop - entry
        signal = {"direction":"short","reason":"BOS down + volume",
                  "entry":round(entry,2),"stop":round(stop,2),
                  "tp1":round(entry-risk,2),"tp2":round(entry-2*risk,2),
                  "bos_time":str(ts)}
        break

report = {
  "symbol":SYMBOL, "day":str(day), "timeframe":INTERVAL,
  "orb":{"start":str(orb_start), "end":str(orb_end), "high":orb_high, "low":orb_low},
  "signal":signal
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/final_trader_report.json","w") as f: json.dump(report,f,indent=2)

print("\n=== FINAL TRADER REPORT ===")
print(f"{SYMBOL} {day} (ET)  TF {INTERVAL}")
print(f"ORB {OPEN}-{(orb_end).strftime('%H:%M')}  High={orb_high:.2f}  Low={orb_low:.2f}")
if signal:
    print(f"{signal['direction'].upper()} | entry {signal['entry']}  stop {signal['stop']}  "
          f"tp1 {signal['tp1']}  tp2 {signal['tp2']}  [{signal['reason']}] at {signal['bos_time']}")
else:
    print("No qualifying setup (no BOS+volume).")
print("Saved: outputs/final_trader_report.json\n")

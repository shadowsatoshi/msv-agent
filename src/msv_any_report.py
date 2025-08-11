import argparse, json
from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf

# ----- sensible defaults per Yahoo suffix (you can override with flags) -----
EX_DEFAULTS = {
  ".L":  ("Europe/London",       "08:00", "16:30"),
  ".TO": ("America/Toronto",     "09:30", "16:00"),
  ".NS": ("Asia/Kolkata",        "09:15", "15:30"),
  ".BO": ("Asia/Kolkata",        "09:15", "15:30"),
  ".AX": ("Australia/Sydney",    "10:00", "16:00"),
  ".HK": ("Asia/Hong_Kong",      "09:30", "16:00"),
  ".KS": ("Asia/Seoul",          "09:00", "15:30"),
  ".TW": ("Asia/Taipei",         "09:00", "13:30"),
  ".T":  ("Asia/Tokyo",          "09:00", "15:00"),
  ".PA": ("Europe/Paris",        "09:00", "17:30"),
  ".F":  ("Europe/Berlin",       "09:00", "17:30"),
  ".MI": ("Europe/Rome",         "09:00", "17:30"),
  ".AS": ("Europe/Amsterdam",    "09:00", "17:40"),
  ".SW": ("Europe/Zurich",       "09:00", "17:30"),
  ".VX": ("Europe/Zurich",       "09:00", "17:30"),
  ".SA": ("America/Sao_Paulo",   "10:00", "17:30"),
  ".SS": ("Asia/Shanghai",       "09:30", "15:00"),
  ".SZ": ("Asia/Shanghai",       "09:30", "15:00"),
}
US_TZ, US_OPEN, US_CLOSE = "America/New_York", "09:30", "16:00"  # default for SPY, AAPL, futures like ES=F (override if you want RTH vs. globex)

def infer_defaults(symbol):
    for suf, vals in EX_DEFAULTS.items():
        if symbol.endswith(suf):
            return vals
    # futures "=F" or US tickers
    return (US_TZ, US_OPEN, US_CLOSE)

def atr(series_h, series_l, series_c, n=14):
    pc = series_c.shift(1)
    tr = pd.concat([(series_h-series_l),
                    (series_h-pc).abs(),
                    (series_l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def run(symbol, tz=None, market_open=None, market_close=None,
        interval="5m", period="10d", orb_minutes=30,
        vol_mult=1.5, atr_buf_mult=0.25):
    # defaults by suffix
    d_tz, d_open, d_close = infer_defaults(symbol)
    tz = tz or d_tz
    market_open = market_open or d_open
    market_close = market_close or d_close

    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, prepost=False)
    if df.empty:
        raise SystemExit(f"No data for {symbol}. Check the symbol/suffix (e.g., VOD.L, 9988.HK).")

    # handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]  # Open, High, Low, Close, Adj Close, Volume

    # localize to market timezone
    df.index = (df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz))
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # choose latest day with session data
    day = df.index[-1].date()
    day_df = df[df.index.date == day].between_time(market_open, market_close)
    if day_df.empty:
        day = (df.index[-1] - pd.Timedelta(days=1)).date()
        day_df = df[df.index.date == day].between_time(market_open, market_close)
        if day_df.empty:
            raise SystemExit(f"No regular session found for {symbol} in {period}.")

    # indicators
    a = atr(day_df["High"], day_df["Low"], day_df["Close"], 14)

    # ORB
    orb_start = day_df.index[0]
    orb_end   = orb_start + pd.Timedelta(minutes=orb_minutes)
    win = day_df[(day_df.index>=orb_start) & (day_df.index<orb_end)]
    orb_high, orb_low = float(win["High"].max()), float(win["Low"].min())

    # swings (causal rolling extremes)
    roll = 9
    swing_high = day_df["High"].rolling(roll).max().shift(1)
    swing_low  = day_df["Low"].rolling(roll).min().shift(1)

    # post-ORB first BOS with ATR buffer + volume confirmation
    post = day_df[day_df.index >= orb_end].copy()
    med_vol = post["Volume"].rolling(20).median()

    signal = None
    for ts, row in post.iterrows():
        buf = max(atr_buf_mult * (a.loc[ts] if ts in a.index else 0), 0.0005 * row["Close"])  # min 5bp
        sh, sl = swing_high.loc[ts], swing_low.loc[ts]
        mv = med_vol.loc[ts] if pd.notna(med_vol.loc[ts]) else row["Volume"]
        vol_ok = row["Volume"] >= vol_mult * mv

        if pd.notna(sh) and row["Close"] > sh + buf and vol_ok:
            entry, stop = orb_high, orb_low
            risk = entry - stop
            signal = {"direction":"long","reason":"BOS up + volume",
                      "entry":round(entry,2), "stop":round(stop,2),
                      "tp1":round(entry+risk,2), "tp2":round(entry+2*risk,2),
                      "bos_time":str(ts)}
            break

        if pd.notna(sl) and row["Close"] < sl - buf and vol_ok:
            entry, stop = orb_low, orb_high
            risk = stop - entry
            signal = {"direction":"short","reason":"BOS down + volume",
                      "entry":round(entry,2), "stop":round(stop,2),
                      "tp1":round(entry-risk,2), "tp2":round(entry-2*risk,2),
                      "bos_time":str(ts)}
            break

    report = {
      "symbol": symbol, "day": str(day), "timeframe": interval,
      "tz": tz, "session_open": market_open, "session_close": market_close,
      "orb": {"start": str(orb_start), "end": str(orb_end), "high": orb_high, "low": orb_low},
      "signal": signal
    }

    Path("outputs").mkdir(exist_ok=True)
    out = f"outputs/final_trader_report_{symbol.replace('=','').replace('.','_')}_{day}.json"
    with open(out, "w") as f: json.dump(report, f, indent=2)
    # console print
    print("\n=== FINAL TRADER REPORT ===")
    print(f"{symbol} {day}  TF {interval}  ({tz})")
    print(f"ORB {market_open}-{pd.Timestamp(orb_end).strftime('%H:%M')}  High={orb_high:.2f}  Low={orb_low:.2f}")
    if signal:
        print(f"{signal['direction'].upper()} | entry {signal['entry']}  stop {signal['stop']}  "
              f"tp1 {signal['tp1']}  tp2 {signal['tp2']}  [{signal['reason']}] at {signal['bos_time']}")
    else:
        print("No qualifying setup (no BOS+volume).")
    print(f"Saved: {out}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze ANY ticker with ORB + BOS + Volume.")
    p.add_argument("--symbol", required=True, help="e.g., SPY, AAPL, VOD.L, 9988.HK, RELIANCE.NS, ES=F")
    p.add_argument("--tz", help="IANA tz like Europe/London, Asia/Hong_Kong (auto by suffix if omitted)")
    p.add_argument("--open", dest="open_", help="Session open HH:MM (auto by suffix if omitted)")
    p.add_argument("--close", dest="close_", help="Session close HH:MM (auto by suffix if omitted)")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="10d")
    p.add_argument("--orb", type=int, default=30, help="Opening range minutes (default 30)")
    p.add_argument("--vol_mult", type=float, default=1.5)
    p.add_argument("--atr_buf", type=float, default=0.25)
    args = p.parse_args()
    run(args.symbol, args.tz, args.open_, args.close_, args.interval, args.period, args.orb, args.vol_mult, args.atr_buf)

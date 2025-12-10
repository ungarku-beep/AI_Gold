import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5    # Dataset lebih rapi & stabil
BARS = 5000                     # panjang history
OUTFILE = "data/history.csv"

def fetch_history():
    print("[HISTORY] Connecting to MT5...")

    if not mt5.initialize():
        raise RuntimeError("MT5 init failed!")

    mt5.symbol_select(SYMBOL)

    print("[HISTORY] Downloading candles...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

    df = pd.DataFrame(rates)

    # Tambah indikator sederhana
    df["MA"] = df["close"].rolling(20).mean()
    df["ATR"] = abs(df["high"] - df["low"]).rolling(14).mean()

    df = df.rename(columns={
        "open":"Open",
        "high":"High",
        "low":"Low",
        "close":"Close"
    })

    df.dropna(inplace=True)

    df.to_csv(OUTFILE, index=False)
    print(f"[HISTORY] Saved: {OUTFILE}")

if __name__ == "__main__":
    fetch_history()

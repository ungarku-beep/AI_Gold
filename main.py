# main.py (REAL TRADING + AI + SCALER FIXED)
"""
Real-trading version:
 - Loads model + scaler from model.pkl (dictionary)
 - Works with new trainer.py
 - Safe for demo account
"""

import os
import time
import math
import csv
import traceback
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import joblib

# import ai_core
try:
    import ai_core as ai_core_mod
    AI_AVAILABLE = True
except Exception as e:
    print("[AI] ai_core import error:", e)
    AI_AVAILABLE = False
    ai_core_mod = None


# ================================
# CONFIG
# ================================
CONFIG = {
    "SYMBOL": "XAUUSD",
    "TIMEFRAME_PRIMARY": mt5.TIMEFRAME_M5,
    "TIMEFRAME_CONFIRM": mt5.TIMEFRAME_M30,
    "TIMEFRAME_TREND": mt5.TIMEFRAME_H1,

    "ATR_PERIOD": 14,
    "ATR_MULTIPLIER_SL": 3.0,
    "ATR_MULTIPLIER_TP": 1.5,
    "MIN_ATR": 0.01,

    "MA_PERIOD_TREND": 50,
    "MAX_POSITIONS": 5,

    "FIXED_LOT": 0.01,
    "RISK_PERCENT": 1.0,

    "MIN_LOT": 0.01,
    "MAX_LOT": 5.0,

    "DEVIATION": 10,
    "ORDER_FILLING": mt5.ORDER_FILLING_IOC,

    "CHECK_INTERVAL": 5,
    "REAL_TRADING": True,

    "NEWS_CSV": "news.csv",
    "NEWS_WINDOW_MINUTES": 30,
    "HIGH_IMPACT_TAGS": ["high", "High", "H"],

    "LOG_FILE": "python_ea.log",
    "MAX_DAILY_TRADES": 50,
    "MIN_BALANCE": 10.0,

    # AI thresholds
    "BUY_PROB_THRESHOLD": 25.0,
    "SELL_PROB_THRESHOLD": 25.0,

    "MODEL_PATH": os.path.join("model", "model.pkl"),
}


# ================================
# LOGGING
# ================================
def log(*args):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    line = f"[{ts} UTC] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


# ================================
# MT5 HELPERS
# ================================
def ensure_mt5_connected():
    if not mt5.initialize():
        log("[MT5] initialize failed:", mt5.last_error())
        time.sleep(1)
        if not mt5.initialize():
            log("[MT5] initialize failed again:", mt5.last_error())
            return False
    acct = mt5.account_info()
    if acct is None:
        log("[MT5] account_info() returned None")
        return False
    log("[MT5] Connected:", acct.login, acct.server)
    return True


def get_rates(symbol, timeframe, n):
    try:
        return mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    except Exception as e:
        log("[MT5] rates error:", e)
        return None


def compute_atr_from_rates(rates, period=14):
    if rates is None or len(rates) < period + 2:
        return None
    highs = np.array([r['high'] for r in rates])
    lows = np.array([r['low'] for r in rates])
    closes = np.array([r['close'] for r in rates])

    trs = []
    for i in range(1, len(rates)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)

    if len(trs) < period:
        return float(np.mean(trs))
    return float(np.mean(trs[-period:]))


def simple_ma_from_rates(rates, period=50):
    if rates is None or len(rates) < period:
        return None
    closes = np.array([r['close'] for r in rates])
    return float(np.mean(closes[-period:]))


# ================================
# NEWS FILTER
# ================================
def read_news_csv(path):
    entries = []
    if not path or not os.path.exists(path):
        return entries
    try:
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dt_str = row.get("datetime") or row.get("date") or row.get("time")
                if not dt_str:
                    continue

                dt = None
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except:
                        pass
                if dt is None:
                    try:
                        dt = datetime.fromisoformat(dt_str).astimezone(timezone.utc)
                    except:
                        continue

                entries.append({
                    "datetime": dt,
                    "impact": row.get("impact", ""),
                    "symbol": row.get("symbol", "").strip()
                })
    except Exception as e:
        log("[NEWS] read error:", e)
    return entries


def is_news_nearby(news_entries, symbol):
    if not news_entries:
        return False
    now = datetime.now(timezone.utc)
    window = timedelta(minutes=CONFIG["NEWS_WINDOW_MINUTES"])
    for e in news_entries:
        if any(tag in str(e["impact"]) for tag in CONFIG["HIGH_IMPACT_TAGS"]):
            if abs((e["datetime"] - now).total_seconds()) <= window.total_seconds():
                return True
    return False


# ================================
# MODEL + SCALER LOADING (FIXED)
# ================================
MODEL = None
SCALER = None

def load_model_and_scaler():
    global MODEL, SCALER

    path = CONFIG["MODEL_PATH"]

    if not os.path.exists(path):
        log("[MODEL] NOT FOUND at", path)
        MODEL = None
        SCALER = None
        return

    try:
        pkg = joblib.load(path)
        log("[MODEL] Loaded model package:", path)

        if isinstance(pkg, dict):
            MODEL = pkg.get("model", None)
            SCALER = pkg.get("scaler", None)
            log("[MODEL] Model + scaler loaded from dict.")
        else:
            MODEL = pkg
            SCALER = None
            log("[MODEL] Loaded single model (no scaler).")
    except Exception as e:
        log("[MODEL] load error:", e)
        MODEL = None
        SCALER = None


# ================================
# AI SIGNAL
# ================================
def get_ai_signal(symbol):
    if MODEL is None or not AI_AVAILABLE:
        return None, 0.0

    try:
        features = ai_core_mod.build_features_6(symbol)

        pred, prob, direction = ai_core_mod.predict_signal(MODEL, SCALER, features)

        if prob < 1.0:  
            prob *= 100.0

        log("[AI] pred:", pred, "prob:", prob, "dir:", direction)

        if direction == "BUY" and prob >= CONFIG["BUY_PROB_THRESHOLD"]:
            return "BUY", prob

        if direction == "SELL" and prob >= CONFIG["SELL_PROB_THRESHOLD"]:
            return "SELL", prob

        return None, prob

    except Exception as e:
        log("[AI] exception:", e)
        return None, 0.0


# ================================
# POSITIONS & ORDERS
# ================================
def get_open_positions(symbol):
    try:
        pos = mt5.positions_get(symbol=symbol)
        return pos if pos else []
    except:
        return []


def send_order(direction, volume, price, sl, tp, symbol):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": CONFIG["DEVIATION"],
        "magic": 202504,
        "comment": "PY_AI_EA",
        "type_filling": CONFIG["ORDER_FILLING"]
    }

    log("[ORDER] sending:", req)

    if not CONFIG["REAL_TRADING"]:
        log("[SIMULATED ORDER] OK:", req)
        return {"sim": True}

    try:
        res = mt5.order_send(req)
        log("[ORDER] result:", res)
        return res
    except Exception as e:
        log("[ORDER] exception:", e)
        return None


# ================================
# TRAILING STOP
# ================================
def manage_open_positions(symbol):
    positions = get_open_positions(symbol)
    if not positions:
        return

    tick = mt5.symbol_info_tick(symbol)

    for pos in positions:
        atr_rates = get_rates(symbol, CONFIG["TIMEFRAME_PRIMARY"], CONFIG["ATR_PERIOD"] + 3)
        atr = compute_atr_from_rates(atr_rates, CONFIG["ATR_PERIOD"])
        if atr is None:
            continue

        distance = atr * 1.0
        current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - distance
            if new_sl > pos.sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp
                })
        else:
            new_sl = current_price + distance
            if pos.sl == 0 or new_sl < pos.sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp
                })


# ================================
# MAIN LOOP
# ================================
def main_loop():
    if not ensure_mt5_connected():
        return

    symbol = CONFIG["SYMBOL"]
    news_entries = read_news_csv(CONFIG["NEWS_CSV"])

    trades_today = 0
    day_start = datetime.now(timezone.utc).date()

    while True:
        try:
            # reset counter per hari
            if datetime.now(timezone.utc).date() != day_start:
                day_start = datetime.now(timezone.utc).date()
                trades_today = 0

            ensure_mt5_connected()

            acct = mt5.account_info()
            if acct is None or acct.balance < CONFIG["MIN_BALANCE"]:
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            if is_news_nearby(news_entries, symbol):
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            direction, score = get_ai_signal(symbol)
            log("[SIGNAL] direction:", direction, "score:", score)

            positions = get_open_positions(symbol)
            if len(positions) >= CONFIG["MAX_POSITIONS"]:
                manage_open_positions(symbol)
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            if direction is None:
                manage_open_positions(symbol)
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            # prepare entry
            rates = get_rates(symbol, CONFIG["TIMEFRAME_PRIMARY"], CONFIG["ATR_PERIOD"] + 5)
            atr = compute_atr_from_rates(rates, CONFIG["ATR_PERIOD"])

            if atr is None or atr < CONFIG["MIN_ATR"]:
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            tick = mt5.symbol_info_tick(symbol)
            entry_price = tick.ask if direction == "BUY" else tick.bid

            sl = entry_price - atr * CONFIG["ATR_MULTIPLIER_SL"] if direction == "BUY" else entry_price + atr * CONFIG["ATR_MULTIPLIER_SL"]
            tp = entry_price + atr * CONFIG["ATR_MULTIPLIER_TP"] if direction == "BUY" else entry_price - atr * CONFIG["ATR_MULTIPLIER_TP"]

            lot = CONFIG["FIXED_LOT"]

            log(f"[MAIN] ENTRY dir={direction} lot={lot} price={entry_price} sl={sl} tp={tp}")

            res = send_order(direction, lot, entry_price, sl, tp, symbol)

            if res is None:
                log("[MAIN] order error")
            else:
                trades_today += 1

            manage_open_positions(symbol)
            time.sleep(CONFIG["CHECK_INTERVAL"])

        except KeyboardInterrupt:
            log("[MAIN] EXIT by user")
            break

        except Exception as e:
            log("[MAIN LOOP ERROR]", e)
            log(traceback.format_exc())
            time.sleep(5)


# ================================
# ENTRYPOINT
# ================================
if __name__ == "__main__":
    log("[START] Python EA starting. REAL_TRADING =", CONFIG["REAL_TRADING"])
    load_model_and_scaler()
    main_loop()

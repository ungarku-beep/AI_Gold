# main_ma50_final.py
# Aggressive EA + MA50 H1 trend filter + dynamic lot + multi-entry + dynamic TS
# WARNING: This script can perform real trading when CONFIG["REAL_TRADING"] = True.
# Test on demo account first.

import os
import time
import math
import csv
import traceback
from datetime import datetime, timedelta, timezone
import subprocess
from collections import deque, defaultdict

import MetaTrader5 as mt5
import numpy as np
import joblib

# optional ai_core import
try:
    import ai_core as ai_core_mod
    AI_AVAILABLE = True
except Exception as e:
    print("[AI] ai_core import error:", e)
    AI_AVAILABLE = False
    ai_core_mod = None

# ================================
# CONFIG (default values)
# ================================
CONFIG = {
    # Mode
    "AGGRESSIVE_MODE": True,                # True = direct entry on pred, False = use thresholds

    # Prediction smoothing
    "PRED_SMOOTH_WINDOW": 5,
    "PROB_MULTIPLIER": 1.2,

    # symbol/timeframes
    "SYMBOL": "XAUUSD",
    "TIMEFRAME_PRIMARY": mt5.TIMEFRAME_M5,
    "TIMEFRAME_CONFIRM": mt5.TIMEFRAME_M30,
    "TIMEFRAME_TREND": mt5.TIMEFRAME_H1,

    # ATR / SL TP
    "ATR_PERIOD": 14,
    "ATR_MULTIPLIER_SL": 3.0,
    "ATR_MULTIPLIER_TP": 1.5,
    "MIN_ATR": 0.01,

    "MA_PERIOD_TREND": 50,
    "MAX_POSITIONS": 10,
    "MIN_OPEN_POSITIONS": 2,

    # lot / risk
    "FIXED_LOT": 0.01,
    "RISK_PERCENT": 1.0,
    "MIN_LOT": 0.01,
    "MAX_LOT": 5.0,

    "DEVIATION": 10,
    "ORDER_FILLING": mt5.ORDER_FILLING_IOC,

    "CHECK_INTERVAL": 5,
    "REAL_TRADING": True,

    # news / safety
    "NEWS_CSV": "news.csv",
    "NEWS_WINDOW_MINUTES": 30,
    "HIGH_IMPACT_TAGS": ["high", "High", "H"],

    "LOG_FILE": "python_ea.log",
    "MAX_DAILY_TRADES": 50,
    "MIN_BALANCE": 10.0,

    # AI thresholds (used only if AGGRESSIVE_MODE = False)
    "BUY_PROB_THRESHOLD": 62.0,
    "SELL_PROB_THRESHOLD": 62.0,

    "MODEL_PATH": os.path.join("model", "model.pkl"),

    # Aggressive-mode safeguards
    "MAX_DAILY_TRADES_AGGR": 20,
    "MAX_PER_HOUR_AGGR": 5,

    # Trailing stop dynamics
    "TRAILING_ATR_MULTIPLIER": 1.0,

    # Misc
    "AUTO_PUSH_CHANGES": False,
    "GIT_COMMIT_MESSAGE": "chore(ea): update aggressive-mode config/code",
}

# Prediction buffers (per symbol)
PRED_BUFFER = defaultdict(lambda: deque(maxlen=CONFIG["PRED_SMOOTH_WINDOW"]))

# ================================
# EXTERNAL CONFIG OVERRIDE (config.py)
# ================================
def apply_external_config_override():
    cfg_path = os.path.join(os.getcwd(), "config.py")
    if not os.path.exists(cfg_path):
        return
    try:
        ns = {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, ns)
        for k, v in ns.items():
            if k in CONFIG and not k.startswith("__"):
                CONFIG[k] = v
        log("[CONFIG] Loaded overrides from config.py")
    except Exception as e:
        log("[CONFIG] Failed to load config.py:", e)

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
# GITHUB SYNC + OPTIONAL PUSH HELPERS
# ================================
def sync_with_github():
    try:
        log("[SYNC] Starting git pull to fetch latest model/code...")
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True, timeout=60)
        out = result.stdout or result.stderr
        if "Already up to date" not in out and "Already up-to-date" not in out:
            log("[SYNC] Git pull successful. Changes fetched.")
        else:
            log("[SYNC] Git pull successful. Repository already up-to-date.")
    except subprocess.CalledProcessError as e:
        log("[SYNC] ERROR: Git pull failed. Ensure Git is installed and repository is set up.")
        log(f"Error output: {e.stderr.strip() if e.stderr else e.stdout}")
    except FileNotFoundError:
        log("[SYNC] ERROR: 'git' command not found. Ensure Git is in your system PATH.")
    except Exception as e:
        log(f"[SYNC] An unknown error occurred during sync: {e}")

def push_changes_to_github(commit_message=None):
    if not CONFIG.get("AUTO_PUSH_CHANGES", False):
        log("[GIT] AUTO_PUSH_CHANGES disabled. Skipping git push.")
        return False
    commit_message = commit_message or CONFIG.get("GIT_COMMIT_MESSAGE", "update")
    try:
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True, text=True, timeout=30)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True, timeout=30)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True, capture_output=True, text=True, timeout=60)
        log("[GIT] Changes pushed to origin/main.")
        return True
    except subprocess.CalledProcessError as e:
        log("[GIT] Push failed:", e.stderr or e.stdout)
        return False
    except Exception as e:
        log("[GIT] Push exception:", e)
        return False

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
# MODEL + SCALER LOADING
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
# PREDICTION AGGREGATION
# ================================
def aggregate_predictions(symbol, pred, prob):
    buf = PRED_BUFFER[symbol]
    buf.append((int(pred), float(prob)))
    votes = [p for p, _ in buf]
    avg_prob = np.mean([pr for _, pr in buf]) if len(buf) > 0 else prob
    ones = votes.count(1)
    zeros = votes.count(0)
    direction = "BUY" if ones >= zeros else "SELL"
    agg_prob = float(avg_prob) * float(CONFIG.get("PROB_MULTIPLIER", 1.0))
    agg_prob = min(100.0, agg_prob)
    return direction, agg_prob

def get_ai_signal(symbol):
    if MODEL is None or not AI_AVAILABLE:
        return None, 0.0
    try:
        features = ai_core_mod.build_features_6(symbol)
        pred, prob, dir_from_model = ai_core_mod.predict_signal(MODEL, SCALER, features)
        if prob < 1.0:
            prob *= 100.0
        agg_dir, agg_prob = aggregate_predictions(symbol, pred, prob)
        if CONFIG.get("AGGRESSIVE_MODE", False):
            log("[AI] (AGGREGATED) pred:", pred, "prob:", prob, "agg_dir:", agg_dir, "agg_prob:", agg_prob)
            return agg_dir, agg_prob
        else:
            log("[AI] pred:", pred, "prob:", prob, "dir_model:", dir_from_model)
            if dir_from_model == "BUY" and prob >= CONFIG.get("BUY_PROB_THRESHOLD", 62.0):
                return "BUY", prob
            if dir_from_model == "SELL" and prob >= CONFIG.get("SELL_PROB_THRESHOLD", 62.0):
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
# TRAILING STOP (dynamic ATR)
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
        trail_distance = atr * CONFIG.get("TRAILING_ATR_MULTIPLIER", 1.0)
        current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        try:
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = current_price - trail_distance
                if new_sl > pos.sl:
                    mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    })
                    log(f"[TS] Updated BUY pos {pos.ticket} SL -> {new_sl}")
            else:
                new_sl = current_price + trail_distance
                if pos.sl == 0 or new_sl < pos.sl:
                    mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    })
                    log(f"[TS] Updated SELL pos {pos.ticket} SL -> {new_sl}")
        except Exception as e:
            log("[TS] update error:", e)

# ================================
# MAIN LOOP
# ================================
def main_loop():
    if not ensure_mt5_connected():
        return

    symbol = CONFIG["SYMBOL"]
    news_entries = read_news_csv(CONFIG["NEWS_CSV"])

    trades_today = 0
    trades_this_hour = 0
    hour_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    day_start = datetime.now(timezone.utc).date()

    while True:
        try:
            now = datetime.now(timezone.utc)
            # reset counters
            if now.date() != day_start:
                day_start = now.date()
                trades_today = 0
            if now >= hour_start + timedelta(hours=1):
                hour_start = now.replace(minute=0, second=0, microsecond=0)
                trades_this_hour = 0

            ensure_mt5_connected()

            acct = mt5.account_info()
            if acct is None or acct.balance < CONFIG["MIN_BALANCE"]:
                log("[MAIN] Account not ready or below MIN_BALANCE. Sleeping.")
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            if is_news_nearby(news_entries, symbol):
                log("[MAIN] News nearby. Skipping cycle.")
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            # allow live override via config.py
            apply_external_config_override()

            # get AI signal (aggregated)
            direction, score = get_ai_signal(symbol)
            log("[SIGNAL] direction:", direction, "score:", score)

            # ================================
            # TREND FILTER USING MA50 (H1)
            # ================================
            if direction is not None:
                h1_rates = get_rates(symbol, CONFIG["TIMEFRAME_TREND"], CONFIG["MA_PERIOD_TREND"] + 3)
                ma50 = simple_ma_from_rates(h1_rates, CONFIG["MA_PERIOD_TREND"])
                if ma50 is not None:
                    # read current price for filter decision
                    tick = mt5.symbol_info_tick(symbol)
                    current_price = tick.ask if direction == "BUY" else tick.bid

                    # BUY only when price > MA50
                    if direction == "BUY" and current_price < ma50:
                        log("[FILTER] BUY blocked — price below MA50 H1 (", current_price, "<", ma50, ")")
                        direction = None

                    # SELL only when price < MA50
                    if direction == "SELL" and current_price > ma50:
                        log("[FILTER] SELL blocked — price above MA50 H1 (", current_price, ">", ma50, ")")
                        direction = None

                    # Optional: tight zone around MA skip trades (avoid chop)
                    # zone = ma50 * 0.001  # 0.1% around MA
                    # if abs(current_price - ma50) <= zone:
                    #     log("[FILTER] Price within MA zone — skip")
                    #     direction = None

            # aggressive safeguards counters
            if CONFIG.get("AGGRESSIVE_MODE", False):
                if trades_today >= CONFIG.get("MAX_DAILY_TRADES_AGGR", CONFIG["MAX_DAILY_TRADES"]):
                    log("[SAFETY] Aggressive daily trades limit reached. Skipping.")
                    time.sleep(CONFIG["CHECK_INTERVAL"])
                    continue
                if trades_this_hour >= CONFIG.get("MAX_PER_HOUR_AGGR", 5):
                    log("[SAFETY] Aggressive hourly trades limit reached. Skipping.")
                    time.sleep(CONFIG["CHECK_INTERVAL"])
                    continue

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
                log("[MAIN] ATR invalid or too small. Skipping.")
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            tick = mt5.symbol_info_tick(symbol)
            entry_price = tick.ask if direction == "BUY" else tick.bid

            sl = entry_price - atr * CONFIG["ATR_MULTIPLIER_SL"] if direction == "BUY" else entry_price + atr * CONFIG["ATR_MULTIPLIER_SL"]
            tp = entry_price + atr * CONFIG["ATR_MULTIPLIER_TP"] if direction == "BUY" else entry_price - atr * CONFIG["ATR_MULTIPLIER_TP"]

            # dynamic lot calculation
            acct = mt5.account_info()
            lot = CONFIG["FIXED_LOT"]
            if acct is not None:
                try:
                    symbol_info = mt5.symbol_info(symbol)
                    risk_amount = acct.balance * CONFIG["RISK_PERCENT"] / 100.0
                    sl_distance = abs(sl - entry_price)
                    if sl_distance >= CONFIG["MIN_ATR"]:
                        risk_dollars_per_lot = sl_distance * 100.0
                        calculated_lot = risk_amount / risk_dollars_per_lot
                        min_lot = symbol_info.volume_min
                        max_lot = symbol_info.volume_max
                        lot_step = symbol_info.volume_step
                        lot = max(min_lot, min(max_lot, calculated_lot))
                        lot = round(lot / lot_step) * lot_step
                except Exception as e:
                    log("[LOT] dynamic lot calc error:", e)
            lot = max(CONFIG["MIN_LOT"], lot)

            # Final safety checks
            if trades_today >= CONFIG.get("MAX_DAILY_TRADES", 9999):
                log("[SAFETY] MAX_DAILY_TRADES reached. Skipping trade.")
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue
            if acct.balance < CONFIG["MIN_BALANCE"]:
                log("[SAFETY] Balance below MIN_BALANCE. Skipping trade.")
                time.sleep(CONFIG["CHECK_INTERVAL"])
                continue

            # Determine how many positions to open (respect MIN_OPEN_POSITIONS & MAX_POSITIONS)
            current_positions = get_open_positions(symbol)
            n_current = len(current_positions)
            if n_current < CONFIG.get("MIN_OPEN_POSITIONS", 1):
                to_open = min(CONFIG.get("MIN_OPEN_POSITIONS", 1) - n_current, CONFIG["MAX_POSITIONS"] - n_current)
            else:
                to_open = 1

            log(f"[MAIN] will open {to_open} position(s) to reach min/mode targets (currently {n_current})")
            opened = 0
            for i in range(to_open):
                if n_current + opened >= CONFIG["MAX_POSITIONS"]:
                    log("[MAIN] reached MAX_POSITIONS during multi-open loop")
                    break
                log(f"[MAIN] ENTRY #{i+1} dir={direction} lot={lot} price={entry_price} sl={sl} tp={tp}")
                res = send_order(direction, lot, entry_price, sl, tp, symbol)
                if res is None:
                    log("[MAIN] order error on open attempt", i+1)
                else:
                    opened += 1
                    trades_today += 1
                    trades_this_hour += 1
                    time.sleep(0.5)
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
    apply_external_config_override()
    sync_with_github()
    load_model_and_scaler()
    main_loop()

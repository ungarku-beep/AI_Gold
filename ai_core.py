# ai_core.py  (adapted from your ai_file_server)
import time, json
from collections import deque

recent_signals = deque(maxlen=6)

def mean(xs): return sum(xs)/len(xs) if xs else 0

def calc_rsi_from_closes(closes, period=14):
    if len(closes) < period+1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period+1):
        diff = closes[-i] - closes[-i-1]
        if diff > 0: gains.append(diff)
        else: losses.append(abs(diff))
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain>0 else 50.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return max(0, min(100, rsi))

def detect_candles_pattern(history_closes, history_opens):
    if len(history_closes) < 2:
        return 0, 0.0
    o1, c1 = history_opens[-1], history_closes[-1]
    o0, c0 = history_opens[-2], history_closes[-2]
    if (c0 < o0) and (c1 > o1) and (c1 - o1) > (o0 - c0):
        return 1, 0.9
    if (c0 > o0) and (c1 < o1) and (o1 - c1) > (c0 - o0):
        return -1, 0.9
    body = abs(c1 - o1)
    high = max(c1, o1)
    low = min(c1, o1)
    candle_range = high - low if high != low else 1e-6
    if body < 0.3 * candle_range:
        if c1 > (low + 0.7*candle_range):
            return 1, 0.6
        if c1 < (low + 0.3*candle_range):
            return -1, 0.6
    return 0, 0.0

def score_from_payload(payload, history_cache):
    ma = float(payload.get("MA", 0))
    atr = float(payload.get("ATR", 0.0001))
    bid = float(payload.get("Bid", 0))
    close = float(payload.get("Close", bid))
    openp = float(payload.get("Open", bid))

    price_dist = abs(bid - ma) / (ma if ma else 1)
    trend_strength = int(40 * min(1.0, price_dist / 0.02))
    dir_sign = 1 if bid > ma else (-1 if bid < ma else 0)

    closes = history_cache.get("closes", [])
    rsi = calc_rsi_from_closes(closes + [close], 14)
    rsi_score = 0; rsi_bias = 0
    if rsi < 35:
        rsi_score = int(20 * (35 - rsi) / 35)
        rsi_bias = 1
    elif rsi > 65:
        rsi_score = int(20 * (rsi - 65) / 35)
        rsi_bias = -1

    opens = history_cache.get("opens", [])
    bias, pat_str = detect_candles_pattern(closes + [close], opens + [openp])
    candle_score = int(20 * pat_str)

    vol_rel = atr / (ma if ma else (bid if bid else 1))
    vol_score = min(10, int(10 * min(1.0, vol_rel / 0.001)))

    recent_signals.append(dir_sign + bias + (1 if rsi_bias==1 else -1 if rsi_bias==-1 else 0))
    cons = sum(1 for x in recent_signals if x>0) - sum(1 for x in recent_signals if x<0)
    cons_bonus = 5 if cons > (len(recent_signals)/2) or cons < -(len(recent_signals)/2) else 0

    total_dir = dir_sign + bias + (1 if rsi_bias==1 else -1 if rsi_bias==-1 else 0)
    direction = 'NONE'
    if total_dir > 0: direction = 'BUY'
    elif total_dir < 0: direction = 'SELL'

    raw = trend_strength + rsi_score + candle_score + vol_score + cons_bonus
    score = max(0, min(100, int(raw)))
    details = {
        "trend_strength": trend_strength,
        "rsi": rsi,
        "rsi_score": rsi_score,
        "candle_bias": bias,
        "candle_score": candle_score,
        "vol_score": vol_score,
        "cons_bonus": cons_bonus,
        "direction": direction,
        "ATR": atr
    }
    return score, direction, details

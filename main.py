# main.py
import time, json, os
import requests, joblib                # <-- tambahan
from ai_core import score_from_payload

# ---------- CONFIG ----------
SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
SLEEP = 1.0  # loop delay in seconds
BUY_THRESHOLD = 65
SELL_THRESHOLD = 35

# ------------------ MODEL AUTO-UPDATER (Langkah 3) ------------------
# Ganti ini dengan username dan repo kamu di GitHub:
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/model/model.pkl"

def update_model():
    print("Checking for updated AI model from GitHub...")

    try:
        response = requests.get(GITHUB_MODEL_URL, timeout=10)

        if response.status_code == 200:
            # buat folder model jika belum ada
            if not os.path.exists("model"):
                os.makedirs("model")

            # simpan file model.pkl
            with open("model/model.pkl", "wb") as f:
                f.write(response.content)

            print(">> Model updated successfully from GitHub.")

        else:
            print(f">> Model update skipped. HTTP Status = {response.status_code}")

    except Exception as e:
        print(">> Error updating model:", e)

# --- placeholder broker adapter (implement per your broker API) ---
class BrokerAdapter:
    def __init__(self):
        pass

    def fetch_latest_bar(self, symbol, timeframe):
        # return dict { "MA":..., "ATR":..., "Bid":..., "Ask":..., "Close":..., "Open":... }
        # Implement using broker API or historic store.
        raise NotImplementedError

    def place_order(self, symbol, direction, volume, sl=None, tp=None):
        print(f"PLACE ORDER {direction} {symbol} vol={volume} sl={sl} tp={tp}")

# -------------- simple runtime --------------
def main_loop():
    broker = BrokerAdapter()
    history = {"closes": [], "opens": []}

    while True:
        try:
            payload = broker.fetch_latest_bar(SYMBOL, TIMEFRAME)

            # history buffer
            history["closes"].append(payload.get("Close"))
            history["opens"].append(payload.get("Open"))
            history["closes"] = history["closes"][-500:]
            history["opens"]  = history["opens"][-500:]

            # AI SCORE
            score, direction, details = score_from_payload(payload, history)
            print(f"score={score} dir={direction} atr={details['ATR']}")

            # ---- trading logic ----
            if direction == "BUY" and score >= BUY_THRESHOLD:
                ask = payload.get("Ask")
                atr = details.get("ATR", 0.5)
                sl = ask - atr*0.8
                tp = ask + atr*1.2
                broker.place_order(SYMBOL, "BUY", volume=0.01, sl=sl, tp=tp)

            elif direction == "SELL" and score >= BUY_THRESHOLD:
                bid = payload.get("Bid")
                atr = details.get("ATR", 0.5)
                sl = bid + atr*0.8
                tp = bid - atr*1.2
                broker.place_order(SYMBOL, "SELL", volume=0.01, sl=sl, tp=tp)

        except Exception as e:
            print("ERR main_loop:", e)

        time.sleep(SLEEP)

# ---------------------------------------------
if __name__ == "__main__":
    update_model()    # <-- Auto-update model sebelum trading
    main_loop()

# ===================================================
# ai_core.py — Build features + Predict (FINAL FIX: Capitalization Match)
# ===================================================
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ** DAFTAR NAMA FITUR **
# Daftar ini SEKARANG menggunakan kapitalisasi yang benar (O, H, L, C)
# untuk mencocokkan fitur yang digunakan saat pelatihan scaler/model.
FEATURE_NAMES = ["Open", "High", "Low", "Close", "Return", "ATR"] 

def build_features_6(symbol):
    """
    Mengambil data rate MT5, menghitung fitur, dan mengembalikan 
    Pandas DataFrame 1-baris dengan nama kolom yang berkapitalisasi benar.
    """
    # Ambil 200 M1 rate
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 200)
    df = pd.DataFrame(rates)

    # Hitung fitur
    df["Return"] = df["close"].pct_change()
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()

    df = df.dropna()
    
    if df.empty:
        # Menangani jika data tidak cukup
        return pd.DataFrame([], columns=FEATURE_NAMES)
        
    last = df.iloc[-1]

    # Ambil nilai fitur. Kita menggunakan nama kolom MT5 (huruf kecil) untuk mengakses nilai.
    feature_values = [
        last["open"],   
        last["high"],   
        last["low"],    
        last["close"],  
        last["Return"], # Asumsi Return dan ATR sudah benar (case-sensitive)
        last["ATR"]
    ]
    
    # Kunci perbaikan: DataFrame baru menggunakan nama kolom yang berkapitalisasi
    # ("Open", "High", dll.) yang diharapkan oleh scaler.
    X_new_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    
    return X_new_df


def predict_signal(model, scaler, features_df):
    """
    Menerima model, scaler, dan DataFrame fitur untuk melakukan prediksi.
    """
    
    if features_df.empty:
        return None, 0.0, None

    # Tentukan data yang akan digunakan untuk prediksi
    if scaler is not None:
        # Sklearn sekarang akan melihat DataFrame dengan nama kolom yang cocok.
        features_to_predict = scaler.transform(features_df)
    else:
        # Jika tidak ada scaler, ubah DataFrame ke NumPy Array
        features_to_predict = features_df.values 

    # Prediksi
    pred = model.predict(features_to_predict)[0]
    
    # Asumsi kelas 1 adalah BUY
    prob = float(model.predict_proba(features_to_predict)[0][1] * 100) 
    
    direction = "BUY" if pred == 1 else "SELL"

    return pred, prob, direction

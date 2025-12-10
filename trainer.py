import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = "data/history.csv"
MODEL_PATH = "model/model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Pastikan kolom wajib ada
    required = ["Open","High","Low","Close","MA","ATR"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Buat target (next candle direction)
    df["return"] = df["Close"].pct_change()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

def train_model():
    print("[TRAIN] Loading data...")
    df = load_data()

    features = df[["Open","High","Low","Close","MA","ATR"]]
    labels = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False
    )

    print("[TRAIN] Training model...")
    model = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.08
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    print(f"[TRAIN] Validation Accuracy = {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    print("[TRAIN] Model saved:", MODEL_PATH)

if __name__ == "__main__":
    train_model()

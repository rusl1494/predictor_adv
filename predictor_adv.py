import os
import platform
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import ta
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 60
FEATURES = ["close", "ema30", "ema100", "rsi"]

DATA_CSV = "btc_data_enriched.csv"
SCALER_PATH = "scaler.save"
MODEL_PATH = "lstm_model.h5"
PRED_JSON = "prediction.json"
PRED_LOG = "predictions_log.csv"
DIFF_LOG = "prediction_diff.csv"
REPORT_TXT = "prediction_report.txt"

REQUEST_TIMEOUT = 10


def load_data():
    df = pd.read_csv(DATA_CSV, index_col="time", parse_dates=True)

    # Ensure ATR exists for risk commentary (not part of scaler/features)
    if "atr" not in df.columns:
        df["atr"] = ta.volatility.average_true_range(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df.dropna(inplace=True)
        df.to_csv(DATA_CSV)

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(df[FEATURES])
    return df, scaled, scaler


def fetch_btc_dominance():
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=REQUEST_TIMEOUT).json()
        return round(resp["data"]["market_cap_percentage"]["btc"], 2)
    except Exception as e:
        print(f"BTC.D fetch error: {e}")
        return None


def predict_next_price():
    df, data, scaler = load_data()
    if len(data) < SEQUENCE_LENGTH:
        raise ValueError("Недостаточно данных для предсказания")

    model = load_model(MODEL_PATH)

    # Prepare input
    input_seq = data[-SEQUENCE_LENGTH:]
    input_seq = np.reshape(input_seq, (1, SEQUENCE_LENGTH, len(FEATURES)))

    # Predict (scaled)
    pred_scaled = model.predict(input_seq, verbose=0)

    # Inverse-transform ONLY the close by creating a dummy feature row
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_real = round(scaler.inverse_transform(dummy)[0, 0], 2)

    # Current market snapshot for context
    current = round(df["close"].iloc[-1], 2)
    last = df.iloc[-1]
    rsi = round(last["rsi"], 2)
    ema30 = round(last["ema30"], 2)
    ema100 = round(last["ema100"], 2)
    atr = round(last["atr"], 2)
    atr_pct = round((atr / last["close"]) * 100, 2)

    trend = (
        "Bullish" if (ema30 > ema100 and rsi < 70)
        else "Bearish" if (ema30 < ema100 and rsi > 30)
        else "Neutral"
    )
    risk = (
        "🟢 Низкий риск" if atr_pct < 1
        else "🟡 Средний риск" if atr_pct < 2.5
        else "🔴 Высокий риск"
    )

    btc_d = fetch_btc_dominance()

    delta = round(pred_real - current, 2)
    arrow = "📈" if delta > 0 else "📉"

    explanation_lines = [
        f"📊 BTC Forecast: ${pred_real}",
        f"📈 Trend: {trend}",
        f"RSI={rsi}, EMA30={ema30}, EMA100={ema100}",
        "➕ EMA30 > EMA100 → рост вероятен" if ema30 > ema100 else "➖ EMA30 < EMA100 → давление на цену",
        "🔵 RSI < 70 — рынок не перекуплен" if rsi < 70 else "🔴 RSI > 70 — возможна коррекция",
        "",
    ]

    if btc_d is not None:
        explanation_lines.append(f"🌐 BTC Dominance: {btc_d}%")
        if btc_d > 50:
            explanation_lines.append("📢 Капитал уходит в BTC — альткоины под давлением")
        elif btc_d < 42:
            explanation_lines.append("🚀 Возможен альтсезон — внимание к ETH, SOL и другим")
        else:
            explanation_lines.append("⚖️ Рынок в балансе между BTC и альтами")
        explanation_lines.append("")

    explanation_lines.append(f"💡 Волатильность: ATR={atr} ({atr_pct}%) → {risk}")
    explanation_lines.append("")
    explanation_lines.append(f"📍 Now: ${current} → {arrow} Δ = {delta}")

    explanation = "\n".join(explanation_lines)
    print(explanation)

    # Save artifacts
    with open(PRED_JSON, "w") as f:
        json.dump({"prediction": pred_real}, f)

    with open(PRED_LOG, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()},{pred_real}\n")

    with open(DIFF_LOG, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()},{pred_real},{current},{delta}\n")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(explanation)

    # Optional: send to local trading bot (guarded with timeout)
    payload = {
        "prediction": pred_real,
        "trend": trend,
        "rsi": rsi,
        "ema30": ema30,
        "ema100": ema100,
        "atr_pct": atr_pct,
        "btc_d": btc_d,
    }

    try:
        requests.post("http://localhost:5000/update_prediction", json=payload, timeout=5)
        print("✅ Prediction sent to trading bot")
    except Exception as e:
        print(f"❌ Error sending to bot: {e}")

    return pred_real, explanation


if __name__ == "__main__":
    pred, text = predict_next_price()

    # ⚠️ Remove hard-coded tokens; load from env if you must send Telegram
    # Example (commented out):
    # TG_TOKEN = os.getenv("TG_TOKEN")
    # TG_CHAT_ID = os.getenv("TG_CHAT_ID")
    # if TG_TOKEN and TG_CHAT_ID:
    #     try:
    #         url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    #         resp = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=REQUEST_TIMEOUT)
    #         if resp.status_code != 200:
    #             print(f"Telegram send error: {resp.text}")
    #     except Exception as e:
    #         print(f"Telegram send exception: {e}")

    if platform.system() == "Windows":
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

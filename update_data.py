import pandas as pd
import requests
import ta

# === Загрузка данных с Kraken ===
def fetch_ohlcv_kraken(pair="XXBTZUSD", interval=1440):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    data = response.json()
    if data.get("error"):
        raise Exception(f"Kraken API error: {data['error']}")
    key = next(k for k in data["result"].keys() if k != "last")
    df = pd.DataFrame(data["result"][key], columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

# === Добавление индикаторов ===
def add_features(df):
    df["ema30"] = ta.trend.ema_indicator(df["close"], window=30)
    df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df.dropna(inplace=True)
    return df

# === Основной запуск ===
df = fetch_ohlcv_kraken()
df = add_features(df)
df.to_csv("btc_data_enriched.csv")
print("✅ Data updated: btc_data_enriched.csv")

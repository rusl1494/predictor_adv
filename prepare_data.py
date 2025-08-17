import pandas as pd
import numpy as np
import requests
import ta  # pip install ta

def fetch_ohlcv_kraken(pair="XXBTZUSD", interval=1440):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    data = response.json()
    key = next(k for k in data["result"].keys() if k != "last")
    df = pd.DataFrame(data["result"][key], columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

def add_features(df):
    df["ema30"] = ta.trend.ema_indicator(df["close"], window=30)
    df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_ohlcv_kraken()
    df = add_features(df)
    df.to_csv("btc_data_enriched.csv")
    print("✅ Saved to btc_data_enriched.csv")

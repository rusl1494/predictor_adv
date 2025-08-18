import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# === Константы ===
SEQUENCE_LENGTH = 60
FEATURES = ["close", "ema30", "ema100", "rsi"]

# === Загрузка и масштабирование данных ===
df = pd.read_csv("btc_data_enriched.csv", index_col="time", parse_dates=True)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, "scaler.save")

# === Подготовка X и y ===
X, y = [], []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i - SEQUENCE_LENGTH:i])
    y.append(scaled_data[i][0])  # Целевая переменная — close
X, y = np.array(X), np.array(y)

# === Обучение модели ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1, callbacks=[EarlyStopping(patience=3)])

model.save("lstm_model.h5")
print("✅ Модель обучена и сохранена как lstm_model.h5")

# === Telegram уведомление ===
try:
    msg = f"✅ Модель переобучена\nПоследняя цена: ${round(df['close'].iloc[-1], 2)}\nДлина данных: {len(df)}"
    bots = [{"token": os.getenv("TG_TOKEN"), "chat_id": os.getenv("TG_CHAT_ID")}]
    for bot in bots:
        url = f"https://api.telegram.org/bot{bot['token']}/sendMessage"
        requests.post(url, json={"chat_id": bot["chat_id"], "text": msg})
except Exception as e:
    print(f"⚠️ Ошибка отправки в Telegram: {e}")


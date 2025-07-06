# src/detect.py

import os
import time
import datetime
import joblib
import pandas as pd

MODEL_PATH = 'models/rf_model.pkl'
STREAM_CSV = 'data/stream.csv'  # will simulate a live feed here


def sensor_stream(csv_path: str):
    """Yield one sample at a time from the CSV feed."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        yield row


def main():
    os.makedirs('logs', exist_ok=True)
    model = joblib.load(MODEL_PATH)

    with open('logs/alerts.log', 'a') as log:
        for sample in sensor_stream(STREAM_CSV):
            # Prepare one-row DataFrame (drop any label column)
            X = pd.DataFrame([sample.drop(labels=['label'], errors='ignore')])
            pred = model.predict(X)[0]
            if pred == 1:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                src = sample.get('src_ip', '<unknown>')
                msg = f"{ts} ALERT: IoT attack detected on {src}\n"
                log.write(msg)
                print(msg.strip())
            # Pace the loop (simulate real-time)
            time.sleep(0.1)


if __name__ == '__main__':
    main()

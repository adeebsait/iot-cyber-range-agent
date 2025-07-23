import os
import time
import joblib
import pandas as pd
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO
import sys

# 1. Paths
ROOT         = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
DATA_STREAM  = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

# 2. Flask + SocketIO
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# 3. Load trained models
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

# 4. Detector thread
def detect(queue: Queue):
    print("[detect] thread starting…", file=sys.stderr)
    # get feature names once (all models share same)
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            print("[detect] received sentinel, exiting", file=sys.stderr)
            break

        print(f"[detect] received msg: time_ms={msg.get('time_ms')}", file=sys.stderr)

        # drop extraneous keys
        msg.pop('time_ms', None)
        msg.pop('label',   None)

        # build numeric-only frame
        df = pd.DataFrame([msg], columns=feature_cols)

        # cast votes to plain ints
        votes = {name: int(m.predict(df)[0]) for name, m in MODELS.items()}
        pred  = 1 if sum(votes.values()) >= (len(votes) / 2) else 0

        if pred == 1:
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes':  votes
            }
            print(f"[detect] emitting alert: {alert}", file=sys.stderr)
            socketio.emit('new_alert', alert)

            # log to file
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, 'a') as f:
                f.write(f"{alert}\n")

# 5. Simulator thread
def simulate(queue: Queue, csv_path: str, delay: float = 0.05):
    print(f"[simulate] loading {csv_path}", file=sys.stderr)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        print(f"[simulate] enqueue time_ms={row.get('time_ms')}", file=sys.stderr)
        queue.put(row.to_dict())
        time.sleep(delay)
    print("[simulate] stream complete, sending sentinel", file=sys.stderr)
    queue.put(None)

# 6. Flask route
@app.route('/')
def index():
    return render_template('index.html')

# 7. Entry point
if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=simulate, args=(q, DATA_STREAM), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

import os
import time
import joblib
import pandas as pd
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO
import sys

# Paths
ROOT         = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
DATA_STREAM  = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

# Flask + SocketIO
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load trained models
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

# Counters
model_vote_counters   = {name: 0 for name in MODELS}
ensemble_alert_counter = 0

def detect(queue: Queue):
    global ensemble_alert_counter
    print("[detect] thread starting…", file=sys.stderr)

    # Get feature columns once (shared by all models)
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            print("[detect] stream ended, emitting summary…", file=sys.stderr)
            break

        # Debug log
        print(f"[detect] got msg time_ms={msg.get('time_ms')}", file=sys.stderr)

        # Drop unwanted keys
        msg.pop('time_ms', None)
        msg.pop('label',   None)

        # Build DataFrame with exactly the trained features
        df = pd.DataFrame([msg], columns=feature_cols)

        # Each model votes
        votes = {}
        for name, clf in MODELS.items():
            pred = int(clf.predict(df)[0])
            votes[name] = pred
            model_vote_counters[name] += pred

        # Ensemble majority
        if sum(votes.values()) >= (len(votes) / 2):
            ensemble_alert_counter += 1
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes':  votes
            }
            print(f"[detect] emitting alert: {alert}", file=sys.stderr)
            socketio.emit('new_alert', alert)

            # Log to file
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, 'a') as f:
                f.write(f"{alert}\n")

    # After the loop, emit the run summary
    summary = {
        'model_votes': model_vote_counters,
        'total_alerts': ensemble_alert_counter
    }
    print(f"[detect] run summary: {summary}", file=sys.stderr)
    socketio.emit('run_summary', summary)

def simulate(queue: Queue, csv_path: str, delay: float = 0.05):
    print(f"[simulate] loading CSV from {csv_path}", file=sys.stderr)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        print(f"[simulate] enqueue time_ms={row.get('time_ms')}", file=sys.stderr)
        queue.put(row.to_dict())
        time.sleep(delay)
    print("[simulate] done streaming, sending sentinel", file=sys.stderr)
    queue.put(None)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=simulate, args=(q, DATA_STREAM), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

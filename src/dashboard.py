import os
import time
import json
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

# Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

# Initialise counters (will be updated by detect thread)
model_vote_counters   = {name: 0 for name in MODELS}
ensemble_alert_counter = 0

def load_past_alerts():
    """Read existing JSON alerts from log and rebuild counters."""
    alerts = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    alert = json.loads(line)
                    alerts.append(alert)
                    # update counters
                    for m, v in alert.get('votes', {}).items():
                        model_vote_counters[m] += v
                except json.JSONDecodeError:
                    continue
    # total alerts is simply len(alerts)
    return alerts

@app.route('/')
def index():
    # Load historic alerts & counters
    past_alerts = load_past_alerts()
    summary = {
        'model_votes': model_vote_counters,
        'total_alerts': len(past_alerts)
    }
    return render_template(
        'index.html',
        init_alerts=json.dumps(past_alerts),
        init_summary=json.dumps(summary)
    )

def detect(queue: Queue):
    global ensemble_alert_counter
    print("[detect] starting…", file=sys.stderr)
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            print("[detect] stream ended", file=sys.stderr)
            break

        # drop synthetic fields
        msg.pop('time_ms', None)
        msg.pop('label',   None)

        df = pd.DataFrame([msg], columns=feature_cols)

        votes = {}
        for name, clf in MODELS.items():
            pred = int(clf.predict(df)[0])
            votes[name] = pred
            model_vote_counters[name] += pred

        if sum(votes.values()) >= (len(votes) / 2):
            ensemble_alert_counter += 1
            alert = {
                'time': time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes': votes
            }
            # emit & log
            socketio.emit('new_alert', alert)
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + "\n")

    # once stream ends, emit final summary
    summary = {
        'model_votes': model_vote_counters,
        'total_alerts': ensemble_alert_counter
    }
    socketio.emit('run_summary', summary)

def simulate(queue: Queue, csv_path: str, delay: float = 0.05):
    print(f"[simulate] loading {csv_path}", file=sys.stderr)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        queue.put(row.to_dict())
        time.sleep(delay)
    queue.put(None)

if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=simulate, args=(q, DATA_STREAM), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

import os
import time
import json
import joblib
import csv
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO
from pandas import DataFrame

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
STREAM_CSV   = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

# Clear previous run’s log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
open(LOG_FILE, 'w').close()

# ─── FLASK & SOCKET.IO SETUP ─────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────────
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
    "GBT":      joblib.load(os.path.join(MODEL_DIR, "Gradient_Boosting.pkl")),
    "MLP":      joblib.load(os.path.join(MODEL_DIR, "MLP_Classifier.pkl")),
}

# ─── METRIC COUNTERS ──────────────────────────────────────────────────────────────
model_vote_counters   = {m: 0 for m in MODELS}
model_tp_counters     = {m: 0 for m in MODELS}
total_attacks         = 0
ensemble_alerts       = 0

# ─── UTIL: read existing log for page reloads ─────────────────────────────────────
def load_past_alerts():
    alerts = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return alerts

# ─── ROUTE: serve dashboard ───────────────────────────────────────────────────────
@app.route('/')
def index():
    past_alerts = load_past_alerts()
    summary = {
        'model_votes':     model_vote_counters,
        'detection_rates': {
            m: round((model_tp_counters[m] / total_attacks * 100) if total_attacks else 0, 2)
            for m in MODELS
        },
        'ensemble_alerts': ensemble_alerts,
        'total_attacks':   total_attacks
    }
    return render_template(
        'index.html',
        init_alerts  = json.dumps(past_alerts),
        init_summary = json.dumps(summary)
    )

# ─── WORKER: detect & log ─────────────────────────────────────────────────────────
def detect(q: Queue):
    global total_attacks, ensemble_alerts

    # Pre-fetch feature names
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = q.get()
        if msg is None:
            # emit final detection summary
            summary = {
                'detection_rates': {
                    m: round((model_tp_counters[m] / total_attacks * 100) if total_attacks else 0, 2)
                    for m in MODELS
                },
                'ensemble_alerts': ensemble_alerts,
                'total_attacks': total_attacks
            }
            socketio.emit('detection_summary', summary)
            break

        # record label for TP counting
        label = int(msg.get('label', 0))
        if label == 1:
            total_attacks += 1

        # drop synthetic fields
        msg.pop('time_ms', None)

        # build DataFrame
        df = DataFrame([msg], columns=feature_cols)

        # model votes + TP count
        votes = {}
        for name, clf in MODELS.items():
            pred = int(clf.predict(df)[0])
            votes[name] = pred
            model_vote_counters[name] += pred
            if label == 1 and pred == 1:
                model_tp_counters[name] += 1

        # ensemble majority decision
        if sum(votes.values()) >= (len(votes)/2):
            ensemble_alerts += 1
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes':  votes
            }
            socketio.emit('new_alert', alert)
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + "\n")

# ─── WORKER: read the updated stream.csv ────────────────────────────────────────────
def simulate(q: Queue):
    with open(STREAM_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            q.put(row)
            time.sleep(int(row.get('time_ms', 50)) / 1000.0)
    q.put(None)

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=simulate, args=(q,), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

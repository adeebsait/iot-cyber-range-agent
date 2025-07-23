import os
import time
import json
import joblib
import csv
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
DATA_STREAM  = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

# Clear previous run's log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
open(LOG_FILE, 'w').close()

# ─── FLASK & SOCKET.IO SETUP ─────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

# ─── GLOBAL COUNTERS ──────────────────────────────────────────────────────────────
model_vote_counters   = {name: 0 for name in MODELS}
ensemble_alert_counter = 0

# ─── UTIL: read existing log ──────────────────────────────────────────────────────
def load_past_alerts():
    alerts = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    alert = json.loads(line)
                    alerts.append(alert)
                    for m, v in alert['votes'].items():
                        model_vote_counters[m] += v
                except json.JSONDecodeError:
                    continue
    return alerts

# ─── ROUTE: serve dashboard ───────────────────────────────────────────────────────
@app.route('/')
def index():
    past_alerts = load_past_alerts()
    summary = {
        'model_votes': model_vote_counters,
        'total_alerts': len(past_alerts)
    }
    return render_template(
        'index.html',
        init_alerts  = json.dumps(past_alerts),
        init_summary = json.dumps(summary)
    )

# ─── WORKER: detect & emit ────────────────────────────────────────────────────────
def detect(queue: Queue):
    global ensemble_alert_counter
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            # send end-of-run summary
            socketio.emit('run_summary', {
                'model_votes':   model_vote_counters,
                'total_alerts':  ensemble_alert_counter
            })
            break

        # strip synthetic keys
        msg.pop('time_ms', None)
        msg.pop('label',   None)

        from pandas import DataFrame
        df = DataFrame([msg], columns=feature_cols)

        votes = {}
        for name, clf in MODELS.items():
            pred = int(clf.predict(df)[0])
            votes[name] = pred
            model_vote_counters[name] += pred

        if sum(votes.values()) >= (len(votes) / 2):
            ensemble_alert_counter += 1
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip','<unknown>'),
                'votes':  votes
            }
            # emit to all clients
            socketio.emit('new_alert', alert)
            # append to log
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + "\n")

# ─── WORKER: tail the log (in case) ────────────────────────────────────────────────
def tail_log():
    with open(LOG_FILE, 'r') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                alert = json.loads(line)
            except json.JSONDecodeError:
                continue
            socketio.emit('new_alert', alert)

# ─── WORKER: simulate live stream ─────────────────────────────────────────────────
def simulate(queue: Queue, csv_path: str, delay: float = 0.05):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queue.put(row)
            time.sleep(delay)
    queue.put(None)

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=tail_log, daemon=True).start()
    Thread(target=simulate, args=(q, DATA_STREAM), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

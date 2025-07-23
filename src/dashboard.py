import os
import time
import json
import joblib
import csv
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO

# Paths
ROOT         = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
DATA_STREAM  = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

# Ensure log folder and clear prior run
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
open(LOG_FILE, 'w').close()

# Flask + SocketIO
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your models
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

# Counters
model_vote_counters   = {name: 0 for name in MODELS}
ensemble_alert_counter = 0

def load_past_alerts():
    """Read the existing log so far and rebuild counters and alert list."""
    alerts = []
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

@app.route('/')
def index():
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
    """
    Consume simulated traffic, run ensemble, and write each alert as JSON to the log.
    """
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            # signal end of stream
            break

        # drop synthetic fields
        msg.pop('time_ms', None)
        msg.pop('label',   None)

        from pandas import DataFrame
        df = DataFrame([msg], columns=feature_cols)

        # vote
        votes = {}
        for name, clf in MODELS.items():
            votes[name] = int(clf.predict(df)[0])

        # if majority malicious
        if sum(votes.values()) >= (len(votes)/2):
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip','<unknown>'),
                'votes':  votes
            }
            # append to log
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + "\n")

def tail_log():
    """
    Tail LOG_FILE: whenever a new line appears, read it,
    update counters, and emit 'new_alert' over SocketIO.
    """
    global ensemble_alert_counter
    with open(LOG_FILE, 'r') as f:
        # go to end of file
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

            # update counters
            for m, v in alert['votes'].items():
                model_vote_counters[m] += v
            ensemble_alert_counter += 1

            # emit live
            socketio.emit('new_alert', alert)

@app.route('/summary')
def emit_summary():
    # in case client wants a manual refresh
    return {
        'model_votes': model_vote_counters,
        'total_alerts': ensemble_alert_counter
    }

def simulate(queue: Queue, csv_path: str, delay: float=0.05):
    """
    Stream the CSV line by line (so detect can write alerts incrementally).
    """
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queue.put(row)
            time.sleep(delay)
    queue.put(None)

if __name__ == "__main__":
    q = Queue()
    # start detector (writes to log only)
    Thread(target=detect,   args=(q,), daemon=True).start()
    # start log tailer (reads from log, emits to clients)
    Thread(target=tail_log, daemon=True).start()
    # start simulator
    Thread(target=simulate, args=(q, DATA_STREAM), daemon=True).start()

    socketio.run(app, host='0.0.0.0', port=5000)

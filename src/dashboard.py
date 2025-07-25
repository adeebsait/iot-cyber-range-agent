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
DATA_DIR     = os.path.join(ROOT, 'data', 'ICUDatasetProcessed')
DATA_STREAM  = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR    = os.path.join(ROOT, 'models')
LOG_FILE     = os.path.join(ROOT, 'logs', 'alerts.log')

STREAM_LENGTH      = 200    # total rows to simulate
ATTACK_OVERSAMPLE  = 0.6    # proportion of attacks in the stream

# Ensure log folder & clear prior log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
open(LOG_FILE, 'w').close()

# ─── FLASK & SOCKET.IO SETUP ─────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# ─── LOAD TRAINED MODELS ─────────────────────────────────────────────────────────
MODELS = {
    "RF":       joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":      joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":   joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
    "GBT":      joblib.load(os.path.join(MODEL_DIR, "Gradient_Boosting.pkl")),
    "MLP":      joblib.load(os.path.join(MODEL_DIR, "MLP_Classifier.pkl")),
}

# ─── METRIC COUNTERS ──────────────────────────────────────────────────────────────
model_vote_counters   = {m: 0 for m in MODELS}   # total positive votes
model_tp_counters     = {m: 0 for m in MODELS}   # true positives
total_attacks         = 0
ensemble_alerts       = 0

# ─── UTIL: build a “complex” short stream ────────────────────────────────────────
def make_stream():
    import pandas as pd
    # Load the three raw files
    df_attack = pd.read_csv(os.path.join(DATA_DIR, "Attack.csv"), low_memory=False)
    df_env    = pd.read_csv(os.path.join(DATA_DIR, "environmentMonitoring.csv"), low_memory=False)
    df_pat    = pd.read_csv(os.path.join(DATA_DIR, "patientMonitoring.csv"), low_memory=False)

    # Label them
    df_attack['label'] = 1
    df_env   ['label'] = 0
    df_pat   ['label'] = 0

    # Oversample attacks for harder detection
    n_att = int(STREAM_LENGTH * ATTACK_OVERSAMPLE)
    n_ben = STREAM_LENGTH - n_att
    samp_attack = df_attack.sample(n=min(n_att, len(df_attack)), random_state=42)
    samp_ben    = pd.concat([
        df_env.sample(n=n_ben//2, random_state=1),
        df_pat.sample(n=n_ben - n_ben//2, random_state=2)
    ])

    stream = pd.concat([samp_attack, samp_ben]).sample(frac=1, random_state=42).reset_index(drop=True)
    # Add simple time_ms
    stream.insert(0, 'time_ms', stream.index * 50)
    stream.to_csv(DATA_STREAM, index=False)
    return list(csv.DictReader(open(DATA_STREAM)))

# ─── UTIL: load previous alerts for refresh ───────────────────────────────────────
def load_past_alerts():
    alerts = []
    if os.path.exists(LOG_FILE):
        for line in open(LOG_FILE):
            try:
                alerts.append(json.loads(line))
            except:
                pass
    return alerts

# ─── ROUTE: serve dashboard ───────────────────────────────────────────────────────
@app.route('/')
def index():
    past = load_past_alerts()
    # build summary
    init_summary = {
        'model_votes': model_vote_counters,
        'detection_rates': {
            m: round((model_tp_counters[m] / total_attacks * 100) if total_attacks else 0, 2)
            for m in MODELS
        },
        'ensemble_alerts': ensemble_alerts,
        'total_attacks': total_attacks
    }
    return render_template(
        'index.html',
        init_alerts  = json.dumps(past),
        init_summary = json.dumps(init_summary)
    )

# ─── WORKER: detect & log ─────────────────────────────────────────────────────────
def detect(queue: Queue):
    global total_attacks, ensemble_alerts
    feature_cols = list(next(iter(MODELS.values())).feature_names_in_)

    while True:
        msg = queue.get()
        if msg is None:
            # end-of-run summary
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

        # preserve label before popping
        label = int(msg.get('label', 0))
        if label == 1:
            total_attacks += 1

        # strip synthetic fields
        msg.pop('time_ms', None)

        # build DataFrame
        from pandas import DataFrame
        df = DataFrame([msg])[feature_cols]

        # each model votes
        votes = {}
        for name, clf in MODELS.items():
            pred = int(clf.predict(df)[0])
            votes[name] = pred
            model_vote_counters[name] += pred
            if label == 1 and pred == 1:
                model_tp_counters[name] += 1

        # ensemble majority
        if sum(votes.values()) >= (len(votes)/2):
            ensemble_alerts += 1
            alert = {
                'time':   time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip','<unknown>'),
                'votes':  votes
            }
            socketio.emit('new_alert', alert)
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + "\n")

# ─── WORKER: simulate the shortened complex stream ─────────────────────────────────
def simulate(queue: Queue):
    for row in make_stream():
        queue.put(row)
        time.sleep(0.05)
    queue.put(None)

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = Queue()
    Thread(target=detect,   args=(q,), daemon=True).start()
    Thread(target=simulate, args=(q,), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)

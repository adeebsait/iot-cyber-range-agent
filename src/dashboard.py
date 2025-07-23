import os
import time
import joblib
import pandas as pd
from queue import Queue
from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO

# 1) Flask app points to the correct templates folder
ROOT = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT, 'templates')
DATA_STREAM = os.path.join(ROOT, 'data', 'stream.csv')
MODEL_DIR   = os.path.join(ROOT, 'models')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*")

# 2) Load your four models by exact filenames
MODELS = {
    "RF":      joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":     joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost":joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":  joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

def detect(queue: Queue):
    """Background thread: consume rows, run ensemble, emit alerts."""
    while True:
        msg = queue.get()
        if msg is None:
            break
        # Build numeric feature frame
        df = pd.DataFrame([msg]).select_dtypes(include='number')
        votes = {n: m.predict(df)[0] for n, m in MODELS.items()}
        pred = 1 if sum(votes.values()) >= (len(votes)/2) else 0
        if pred == 1:
            alert = {
                'time': time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes': votes
            }
            socketio.emit('new_alert', alert)
            with open(os.path.join(ROOT, 'logs', 'alerts.log'), 'a') as f:
                f.write(f"{alert}\n")

def simulate(queue: Queue, csv_path: str, delay: float = 0.05):
    """Background thread: read stream.csv and push rows onto the queue."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        queue.put(row.to_dict())
        time.sleep(delay)
    queue.put(None)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    # 3) Create a single shared queue
    q = Queue()

    # 4) Spin up threads for detection & simulation
    Thread(target=detect,      args=(q,), daemon=True).start()
    Thread(target=simulate,    args=(q, DATA_STREAM), daemon=True).start()

    # 5) Start Flask‐SocketIO (runs the event loop)
    socketio.run(app, host='0.0.0.0', port=5000)

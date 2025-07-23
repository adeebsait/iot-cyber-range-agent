# src/detector.py

import os
import time
import joblib
import pandas as pd
from queue import Queue
from multiprocessing import Process, Manager
from flask_socketio import SocketIO

# 1️⃣ Point to the correct models directory
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models')
)

# 2️⃣ Load exactly the files you have
MODELS = {
    # key = logical name, value = loaded model
    "RF":      joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "KNN":     joblib.load(os.path.join(MODEL_DIR, "KNN_(k=5).pkl")),
    "AdaBoost":joblib.load(os.path.join(MODEL_DIR, "AdaBoost.pkl")),
    "LogReg":  joblib.load(os.path.join(MODEL_DIR, "Logistic_Reg.pkl")),
}

def detect(queue: Queue, socketio: SocketIO):
    """
    Consume rows from the simulator, run each through the ensemble,
    and emit 'new_alert' when the majority vote is 1.
    """
    while True:
        msg = queue.get()
        if msg is None:
            break

        # Build a one-row DataFrame and select only numeric cols
        df = pd.DataFrame([msg]).select_dtypes(include='number')

        # Each model casts a vote
        votes = {name: mdl.predict(df)[0] for name, mdl in MODELS.items()}
        vote_sum = sum(votes.values())
        pred = 1 if vote_sum >= (len(votes) / 2) else 0

        if pred == 1:
            alert = {
                'time': time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip', '<unknown>'),
                'votes': votes
            }
            # send to dashboard
            socketio.emit('new_alert', alert)

            # also log to file
            with open("logs/alerts.log", "a") as f:
                f.write(f"{alert}\n")

def start_services():
    mgr = Manager()
    q = mgr.Queue()

    # import the Flask app/socketio from dashboard.py
    from dashboard import app, socketio

    # Start detector in a separate process
    det_proc = Process(target=detect, args=(q, socketio))
    det_proc.daemon = True
    det_proc.start()

    # Now kick off the simulator (reads data/stream.csv)
    from simulator import start_simulator
    start_simulator(q, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "stream.csv")))

    det_proc.join()

if __name__ == "__main__":
    start_services()

# src/detector.py

import os
import time
import joblib
import pandas as pd
from queue import Queue
from multiprocessing import Process, Manager
from flask_socketio import SocketIO

# Load all models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODELS = {
    name: joblib.load(os.path.join(MODEL_DIR, fname))
    for name, fname in [
        ("RF", "Random_Forest.pkl"),
        ("KNN", "KNN_k=5.pkl"),
        ("AdaBoost", "AdaBoost.pkl"),
        ("LogReg", "Logistic_Reg.pkl")
    ]
}

def detect(queue: Queue, socketio: SocketIO):
    """
    Waits for traffic dicts on queue, runs ensemble vote, emits alerts.
    """
    while True:
        msg = queue.get()
        if msg is None:
            break
        # Prepare numeric feature vector
        df = pd.DataFrame([msg]).select_dtypes(include=['number'])
        # Ensemble vote (majority)
        votes = [model.predict(df)[0] for model in MODELS.values()]
        pred = 1 if sum(votes) >= (len(votes)/2) else 0
        if pred == 1:
            alert = {
                'time': time.strftime("%H:%M:%S"),
                'src_ip': msg.get('src_ip'),
                'models': votes
            }
            socketio.emit('new_alert', alert)
        # optional: log to file
        with open("logs/alerts.log", "a") as f:
            f.write(f"{alert}\n")

def start_services():
    mgr = Manager()
    q = mgr.Queue()

    # Set up Flask-SocketIO
    from dashboard import app, socketio
    # Launch detector process
    det_proc = Process(target=detect, args=(q, socketio))
    det_proc.start()
    # Launch simulator (in main thread)
    from simulator import start_simulator
    start_simulator(q, "../data/stream.csv", delay=0.05)
    det_proc.join()

if __name__ == "__main__":
    start_services()

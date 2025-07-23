# src/simulator.py

import time
import pandas as pd
from queue import Queue

def start_simulator(queue: Queue, csv_path: str, delay: float = 0.1):
    """
    Reads the combined CSV and pushes one row at a time into queue.
    """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        queue.put(row.to_dict())
        time.sleep(delay)  # simulate real traffic pacing
    queue.put(None)  # sentinel to signal end of stream

if __name__ == "__main__":
    from multiprocessing import Manager
    mgr = Manager()
    q = mgr.Queue()
    start_simulator(q, "../data/stream.csv")

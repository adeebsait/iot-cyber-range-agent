#!/usr/bin/env python3
"""
Create data/stream.csv for the simulator.

• Adds a synthetic ‘time_ms’ column (0‑based, 50 ms step).
• Keeps the original rows unchanged otherwise.
"""

import os
import pandas as pd
from pathlib import Path
from random import shuffle

DATA_DIR = Path(__file__).parent.parent / 'data' / 'ICUDatasetProcessed'
OUT_FILE = Path(__file__).parent.parent / 'data' / 'stream.csv'

def main():
    dfs = []
    for fname in ['Attack.csv',
                  'environmentMonitoring.csv',
                  'patientMonitoring.csv']:
        csv_path = DATA_DIR / fname
        df = pd.read_csv(csv_path, low_memory=False)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)

    # Shuffle rows deterministically
    master = master.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Add a synthetic timeline column (50 ms per event)
    master.insert(0, 'time_ms', master.index * 50)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(master):,} rows → {OUT_FILE}")

if __name__ == "__main__":
    main()

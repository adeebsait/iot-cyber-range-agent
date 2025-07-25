#!/usr/bin/env python3
"""
Create data/stream.csv for the simulator with “harder” attacks:

• Guarantees at least ATTACK_TARGET attack flows
• Splits attacks into burst vs. slow‑and‑low scanners (using ip.src or synthetic)
• Injects Gaussian noise into a fraction of numeric attack features
• Shuffles, then adds jittered time_ms at STEP_MS intervals
• Fills the rest with benign flows to reach STREAM_LENGTH total
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / 'data' / 'ICUDatasetProcessed'
OUT_FILE        = Path(__file__).parent.parent / 'data' / 'stream.csv'

STREAM_LENGTH   = 5000    # total rows
ATTACK_TARGET   = 2000    # minimum attack rows
STEP_MS         = 50      # base ms between events
NOISE_RATE      = 0.2     # 20% of attack rows get noise
BURST_SHARE     = 0.7     # 70% of scanners are “burst” scanners

def main():
    # 1) Load raw CSVs
    df_attack = pd.read_csv(DATA_DIR / 'Attack.csv',                low_memory=False)
    df_env    = pd.read_csv(DATA_DIR / 'environmentMonitoring.csv', low_memory=False)
    df_pat    = pd.read_csv(DATA_DIR / 'patientMonitoring.csv',     low_memory=False)

    # 2) Label benign vs. attack
    df_attack['label'] = 1
    df_env   ['label'] = 0
    df_pat   ['label'] = 0

    # 3) Sample exactly ATTACK_TARGET attacks (or all if fewer)
    if len(df_attack) >= ATTACK_TARGET:
        attack_sample = df_attack.sample(n=ATTACK_TARGET, random_state=42).reset_index(drop=True)
    else:
        attack_sample = df_attack.copy().reset_index(drop=True)

    # 4) Cast numeric columns to float64 so we can add noise without dtype issues
    num_cols = attack_sample.select_dtypes(include=[np.number]).columns.tolist()
    attack_sample[num_cols] = attack_sample[num_cols].astype(np.float64)

    # 5) Determine scanner ID column (real or synthetic)
    if 'ip.src' in attack_sample.columns:
        id_col = 'ip.src'
    else:
        num_scanners = max(2, len(attack_sample)//100)
        attack_sample['scanner_id'] = np.random.randint(0, num_scanners, size=len(attack_sample))
        id_col = 'scanner_id'

    # 6) Inject Gaussian noise into NOISE_RATE fraction of the attack rows
    noisy_idxs = attack_sample.sample(frac=NOISE_RATE, random_state=2).index
    for col in num_cols:
        σ     = attack_sample[col].std()
        noise = np.random.normal(0, σ * 0.05, size=len(noisy_idxs))
        attack_sample.loc[noisy_idxs, col] = attack_sample.loc[noisy_idxs, col] + noise

    # 7) Split into burst vs. slow scanners
    unique_ids = attack_sample[id_col].unique().tolist()
    random.shuffle(unique_ids)
    cut        = int(len(unique_ids) * BURST_SHARE)
    burst_ids  = set(unique_ids[:cut])
    slow_ids   = set(unique_ids[cut:])

    attack_burst = attack_sample[attack_sample[id_col].isin(burst_ids)]
    attack_slow  = attack_sample[attack_sample[id_col].isin(slow_ids)]
    attack_final = pd.concat([attack_burst, attack_slow]).reset_index(drop=True)

    # 8) Sample benign flows to fill out to STREAM_LENGTH
    benign_needed = STREAM_LENGTH - len(attack_final)
    env_n = benign_needed // 2
    pat_n = benign_needed - env_n
    benign_sample = pd.concat([
        df_env.sample(n=env_n, random_state=1),
        df_pat.sample(n=pat_n, random_state=2)
    ]).reset_index(drop=True)

    # 9) Build master stream: start with burst + benign, then sprinkle slow
    master = pd.concat([attack_burst, benign_sample]).reset_index(drop=True)
    interval = max(1, len(master) // len(attack_slow))
    for idx, row in attack_slow.iterrows():
        insert_at = idx * interval + random.randint(0, interval-1)
        top  = master.iloc[:insert_at]
        bot  = master.iloc[insert_at:]
        master = pd.concat([top, pd.DataFrame([row]), bot]).reset_index(drop=True)

    # 10) Final shuffle
    master = master.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 11) Add jittered time_ms
    times = []
    for i in range(len(master)):
        base   = i * STEP_MS
        jitter = random.uniform(-STEP_MS*0.4, STEP_MS*0.4)
        times.append(max(0, int(base + jitter)))
    master.insert(0, 'time_ms', times)

    # 12) Write out
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(master):,} rows "
          f"({len(attack_final):,} attacks: "
          f"{len(attack_burst):,} burst, {len(attack_slow):,} slow) → {OUT_FILE}")

if __name__ == "__main__":
    main()

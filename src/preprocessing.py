# src/preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_icudataset(
        data_dir: str,
        test_size: float = 0.2,
        random_state: int = 42
):
    """
    Loads Attack, environmentMonitoring and patientMonitoring CSVs,
    labels them (1=attack, 0=benign), merges, shuffles, handles missing
    values, one-hot-encodes 'protocol' if present, and splits into train/test.

    Returns: X_train, X_test, y_train, y_test
    """
    # Build file paths
    attack_fp = os.path.join(data_dir, 'Attack.csv')
    env_fp = os.path.join(data_dir, 'environmentMonitoring.csv')
    pat_fp = os.path.join(data_dir, 'patientMonitoring.csv')

    # Read each file
    df_attack = pd.read_csv(attack_fp)
    df_env = pd.read_csv(env_fp)
    df_pat = pd.read_csv(pat_fp)

    # Assign labels
    df_attack['label'] = 1
    df_env['label'] = 0
    df_pat['label'] = 0

    # Concatenate and shuffle
    df = pd.concat([df_attack, df_env, df_pat], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Fill missing values
    df = df.fillna(0)

    # One-hot-encode protocol column if it exists
    if 'protocol' in df.columns:
        df = pd.get_dummies(df, columns=['protocol'])

    # Split into features and label
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

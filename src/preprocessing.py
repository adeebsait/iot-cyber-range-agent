import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_icudataset(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads the three ICU CSVs, labels them (1=attack, 0=benign), merges,
    shuffles, fills missing values, one-hot-encodes protocol, then
    selects only numeric features (dropping IPs, timestamps, etc.),
    and splits into train/test.
    """
    # 1. Read files
    attack_fp = os.path.join(data_dir, 'Attack.csv')
    env_fp    = os.path.join(data_dir, 'environmentMonitoring.csv')
    pat_fp    = os.path.join(data_dir, 'patientMonitoring.csv')

    df_attack = pd.read_csv(attack_fp, low_memory=False)
    df_env    = pd.read_csv(env_fp,    low_memory=False)
    df_pat    = pd.read_csv(pat_fp,    low_memory=False)

    # 2. Label
    df_attack['label'] = 1
    df_env   ['label'] = 0
    df_pat   ['label'] = 0

    # 3. Combine & shuffle
    df = pd.concat([df_attack, df_env, df_pat], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 4. Fill missing
    df = df.fillna(0)

    # 5. One-hot encode protocol if exists
    if 'protocol' in df.columns:
        df = pd.get_dummies(df, columns=['protocol'])

    # 6. Drop any remaining non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # 7. Split features & label
    X = df_numeric.drop(columns=['label'])
    y = df_numeric['label'].astype(int)

    # 8. Train/test split
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

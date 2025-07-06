import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_prepare_icudataset(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    chunksize: int = 50000
):
    """
    Loads the three ICU CSVs in chunks (showing progress), labels them,
    merges, shuffles, fills missing, one-hot-encodes protocol, drops
    non-numeric columns, and splits into train/test.
    """

    def load_csv_with_progress(path, desc):
        # generator of DataFrame chunks
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        chunks = []
        for chunk in tqdm(reader, desc=desc, unit='rows'):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    # 1. Read each file in chunks
    attack_fp = os.path.join(data_dir, 'Attack.csv')
    env_fp    = os.path.join(data_dir, 'environmentMonitoring.csv')
    pat_fp    = os.path.join(data_dir, 'patientMonitoring.csv')

    df_attack = load_csv_with_progress(attack_fp, 'Loading Attack.csv')
    df_env    = load_csv_with_progress(env_fp,    'Loading environmentMonitoring.csv')
    df_pat    = load_csv_with_progress(pat_fp,    'Loading patientMonitoring.csv')

    # 2. Label
    df_attack['label'] = 1
    df_env   ['label'] = 0
    df_pat   ['label'] = 0

    # 3. Concatenate & shuffle
    print("Merging and shuffling datasets…")
    df = pd.concat([df_attack, df_env, df_pat], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 4. Fill missing values
    print("Filling missing values…")
    df = df.fillna(0)

    # 5. One-hot encode protocol if present
    if 'protocol' in df.columns:
        print("One-hot encoding 'protocol' column…")
        df = pd.get_dummies(df, columns=['protocol'])

    # 6. Drop non-numeric columns
    print("Dropping non-numeric columns…")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric   = df[numeric_cols]

    # 7. Split into features & label
    X = df_numeric.drop(columns=['label'])
    y = df_numeric['label'].astype(int)

    # 8. Train/test split
    print("Splitting data into train and test sets…")
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

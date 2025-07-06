import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocessing import load_and_prepare_icudataset

def main(data_dir: str, model_out: str = 'models/rf_model.pkl'):
    # 1. Load & split
    X_train, X_test, y_train, y_test = load_and_prepare_icudataset(data_dir)

    # 2. Train the RF agent
    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    joblib.dump(model, model_out)

    # 3. Predict & time it
    t1 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    eval_time = time.time() - t1

    # 4. Compute metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    # 5. Throughput & latency
    t2 = time.time()
    model.predict(X_test)
    pred_time = time.time() - t2
    throughput = len(X_test) / pred_time
    latency    = pred_time / len(X_test)

    # 6. Print metrics
    print("\n=== TRAIN & EVALUATION METRICS ===")
    print(f"Training time:       {train_time:.2f} s")
    print(f"Inference time:      {eval_time:.2f} s")
    print(f"Accuracy:            {acc*100:.2f}%")
    print(f"Precision:           {prec*100:.2f}%")
    print(f"Recall:              {rec*100:.2f}%")
    print(f"F1 Score:            {f1*100:.2f}%")
    print(f"ROC AUC:             {auc*100:.2f}%")
    print(f"False Positive Rate: {fpr*100:.2f}%")
    print(f"False Negative Rate: {fnr*100:.2f}%")
    print(f"Throughput:          {throughput:.1f} samples/s")
    print(f"Latency per sample:  {latency*1000:.2f} ms")

    # 7. Baseline comparison
    baselines = {
        "Hussain et al. (2021)": 0.995123,  # 99.5123%
        "Perera et al. (2024)": 0.9955     # 99.55%
    }
    print("\n=== COMPARISON TO BASELINES ===")
    for name, b_acc in baselines.items():
        imp = (acc - b_acc) * 100
        print(f"{name}: {b_acc*100:.4f}%  →  improvement: {imp:.2f} pp")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for IoT ICS Security"
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to ICUDatasetProcessed folder'
    )
    parser.add_argument(
        '--out', default='models/rf_model.pkl',
        help='Where to save the trained model'
    )
    args = parser.parse_args()
    main(args.data_dir, args.out)

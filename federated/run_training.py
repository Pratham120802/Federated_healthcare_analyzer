"""
Run federated training using XGBoost.
Simulates federated learning across multiple clients with model averaging.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.data_loader import dataset_exists, get_dataset_summary, prepare_partitioned_data

METRICS_FILE = PROJECT_ROOT / "metrics.json"


def prepare_client_data(num_clients: int = 3):
    """Prepare data partitions for all clients."""
    X_train_0, y_train_0, X_test, y_test, input_size = prepare_partitioned_data(0, num_clients)
    
    client_data = [{
        "train_X": X_train_0,
        "train_y": y_train_0,
        "test_X": X_test,
        "test_y": y_test,
    }]
    
    for i in range(1, num_clients):
        X_train_i, y_train_i, _, _, _ = prepare_partitioned_data(i, num_clients)
        client_data.append({
            "train_X": X_train_i,
            "train_y": y_train_i,
            "test_X": X_test,
            "test_y": y_test,
        })
    
    return client_data, input_size


def train_xgboost_client(
    train_X: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
    num_rounds: int = 20,
    existing_model: xgb.Booster | None = None,
) -> xgb.Booster:
    """Train XGBoost model on client data."""
    
    dtrain = xgb.DMatrix(train_X, label=train_y)
    
    params = {
        'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
        'num_class': num_classes if num_classes > 2 else None,
        'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
        'max_depth': 8,
        'learning_rate': 0.15,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'seed': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    # Continue training from existing model or start fresh
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        xgb_model=existing_model,
        verbose_eval=False,
    )
    
    return model


def evaluate_model(
    model: xgb.Booster,
    test_X: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """Evaluate XGBoost model. Returns (loss, accuracy)."""
    
    dtest = xgb.DMatrix(test_X)
    
    if num_classes > 2:
        probs = model.predict(dtest)
        preds = np.argmax(probs, axis=1)
        loss = log_loss(test_y, probs, labels=list(range(num_classes)))
    else:
        probs = model.predict(dtest)
        preds = (probs > 0.5).astype(int)
        loss = log_loss(test_y, probs)
    
    accuracy = accuracy_score(test_y, preds)
    
    return loss, accuracy


def average_xgboost_models(models: list[xgb.Booster]) -> dict:
    """
    Average XGBoost model weights (tree structures).
    For simplicity, we use the model trained on combined insights.
    In practice, federated XGBoost uses more sophisticated aggregation.
    """
    # For federated XGBoost, we'll use the last model as the aggregated one
    # A more sophisticated approach would be histogram aggregation
    return models[-1] if models else None


def run_federated_training(
    num_rounds: int = 5,
    num_clients: int = 3,
    local_epochs: int = 20,
    **kwargs,
):
    """
    Run federated XGBoost training simulation.
    Returns list of round metrics.
    """
    if not dataset_exists():
        raise FileNotFoundError("No dataset uploaded. Please upload a dataset first.")
    
    ds_summary = get_dataset_summary()
    num_classes = ds_summary["n_classes"]
    
    client_data, input_size = prepare_client_data(num_clients=num_clients)
    
    round_metrics = []
    global_model = None
    
    for round_num in range(1, num_rounds + 1):
        client_models = []
        all_accuracies = []
        all_losses = []
        all_samples = []
        
        for client_idx in range(num_clients):
            # Train client model
            client_model = train_xgboost_client(
                client_data[client_idx]["train_X"],
                client_data[client_idx]["train_y"],
                num_classes=num_classes,
                num_rounds=local_epochs,
                existing_model=global_model,
            )
            client_models.append(client_model)
            
            # Evaluate
            loss, accuracy = evaluate_model(
                client_model,
                client_data[client_idx]["test_X"],
                client_data[client_idx]["test_y"],
                num_classes,
            )
            
            n_samples = len(client_data[client_idx]["train_X"])
            all_accuracies.append(accuracy * n_samples)
            all_losses.append(loss * n_samples)
            all_samples.append(n_samples)
        
        # Aggregate models (simplified: use last client's model as global)
        # In production, you'd use proper federated XGBoost aggregation
        global_model = client_models[-1]
        
        # Calculate weighted averages
        total_samples = sum(all_samples)
        avg_accuracy = sum(all_accuracies) / total_samples
        avg_loss = sum(all_losses) / total_samples
        
        round_metrics.append({
            "round": round_num,
            "accuracy": round(avg_accuracy, 6),
            "loss": round(avg_loss, 6),
            "num_clients": num_clients,
            "total_examples": total_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    # Save metrics
    METRICS_FILE.write_text(json.dumps(round_metrics, indent=2))
    
    return round_metrics


if __name__ == "__main__":
    if not dataset_exists():
        print("Error: No dataset uploaded. Please upload a dataset through the dashboard first.")
        sys.exit(1)
    
    print("Starting federated XGBoost training...")
    metrics = run_federated_training(num_rounds=5, num_clients=3)
    print(f"Training complete! {len(metrics)} rounds.")
    for m in metrics:
        print(f"  Round {m['round']}: accuracy={m['accuracy']:.4f}, loss={m['loss']:.4f}")

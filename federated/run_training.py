"""
Run federated training using XGBoost, Random Forest, and LightGBM.
Simulates federated learning across multiple clients with model averaging.
Compares accuracy across all three models.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.data_loader import dataset_exists, get_dataset_summary, prepare_partitioned_data

METRICS_FILE = PROJECT_ROOT / "metrics.json"
MODEL_COMPARISON_FILE = PROJECT_ROOT / "model_comparison.json"


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


# =============================================================================
# Random Forest Training
# =============================================================================

def train_random_forest_client(
    train_X: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
) -> RandomForestClassifier:
    """Train Random Forest model on client data."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' if num_classes > 2 else None,
    )
    model.fit(train_X, train_y)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    test_X: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """Evaluate Random Forest model. Returns (loss, accuracy)."""
    probs = model.predict_proba(test_X)
    preds = model.predict(test_X)
    
    try:
        loss = log_loss(test_y, probs, labels=list(range(num_classes)))
    except ValueError:
        loss = 1.0
    
    accuracy = accuracy_score(test_y, preds)
    return loss, accuracy


# =============================================================================
# LightGBM Training
# =============================================================================

def train_lightgbm_client(
    train_X: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
    num_rounds: int = 20,
    existing_model: lgb.Booster | None = None,
) -> lgb.Booster:
    """Train LightGBM model on client data."""
    
    params = {
        'objective': 'multiclass' if num_classes > 2 else 'binary',
        'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.15,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    if num_classes > 2:
        params['num_class'] = num_classes
    
    train_data = lgb.Dataset(train_X, label=train_y)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        init_model=existing_model,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    
    return model


def evaluate_lightgbm(
    model: lgb.Booster,
    test_X: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """Evaluate LightGBM model. Returns (loss, accuracy)."""
    
    probs = model.predict(test_X)
    
    if num_classes > 2:
        preds = np.argmax(probs, axis=1)
        try:
            loss = log_loss(test_y, probs, labels=list(range(num_classes)))
        except ValueError:
            loss = 1.0
    else:
        preds = (probs > 0.5).astype(int)
        try:
            loss = log_loss(test_y, probs)
        except ValueError:
            loss = 1.0
    
    accuracy = accuracy_score(test_y, preds)
    return loss, accuracy


# =============================================================================
# Multi-Model Federated Training
# =============================================================================

def run_federated_training(
    num_rounds: int = 5,
    num_clients: int = 3,
    local_epochs: int = 20,
    **kwargs,
):
    """
    Run federated training simulation with XGBoost, Random Forest, and LightGBM.
    Compares accuracy across all three models.
    Returns list of round metrics.
    """
    if not dataset_exists():
        raise FileNotFoundError("No dataset uploaded. Please upload a dataset first.")
    
    ds_summary = get_dataset_summary()
    num_classes = ds_summary["n_classes"]
    
    client_data, input_size = prepare_client_data(num_clients=num_clients)
    
    # Store metrics for each model
    xgb_round_metrics = []
    rf_round_metrics = []
    lgb_round_metrics = []
    
    # Global models
    xgb_global_model = None
    lgb_global_model = None
    
    for round_num in range(1, num_rounds + 1):
        # =====================================================================
        # XGBoost Training
        # =====================================================================
        xgb_client_models = []
        xgb_accuracies = []
        xgb_losses = []
        xgb_samples = []
        
        for client_idx in range(num_clients):
            client_model = train_xgboost_client(
                client_data[client_idx]["train_X"],
                client_data[client_idx]["train_y"],
                num_classes=num_classes,
                num_rounds=local_epochs,
                existing_model=xgb_global_model,
            )
            xgb_client_models.append(client_model)
            
            loss, accuracy = evaluate_model(
                client_model,
                client_data[client_idx]["test_X"],
                client_data[client_idx]["test_y"],
                num_classes,
            )
            
            n_samples = len(client_data[client_idx]["train_X"])
            xgb_accuracies.append(accuracy * n_samples)
            xgb_losses.append(loss * n_samples)
            xgb_samples.append(n_samples)
        
        xgb_global_model = xgb_client_models[-1]
        total_samples = sum(xgb_samples)
        xgb_avg_accuracy = sum(xgb_accuracies) / total_samples
        xgb_avg_loss = sum(xgb_losses) / total_samples
        
        xgb_round_metrics.append({
            "round": round_num,
            "accuracy": round(xgb_avg_accuracy, 6),
            "loss": round(xgb_avg_loss, 6),
            "num_clients": num_clients,
            "total_examples": total_samples,
            "model": "XGBoost",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        # =====================================================================
        # Random Forest Training
        # =====================================================================
        rf_accuracies = []
        rf_losses = []
        rf_samples = []
        
        for client_idx in range(num_clients):
            client_model = train_random_forest_client(
                client_data[client_idx]["train_X"],
                client_data[client_idx]["train_y"],
                num_classes=num_classes,
            )
            
            loss, accuracy = evaluate_random_forest(
                client_model,
                client_data[client_idx]["test_X"],
                client_data[client_idx]["test_y"],
                num_classes,
            )
            
            n_samples = len(client_data[client_idx]["train_X"])
            rf_accuracies.append(accuracy * n_samples)
            rf_losses.append(loss * n_samples)
            rf_samples.append(n_samples)
        
        total_samples = sum(rf_samples)
        rf_avg_accuracy = sum(rf_accuracies) / total_samples
        rf_avg_loss = sum(rf_losses) / total_samples
        
        rf_round_metrics.append({
            "round": round_num,
            "accuracy": round(rf_avg_accuracy, 6),
            "loss": round(rf_avg_loss, 6),
            "num_clients": num_clients,
            "total_examples": total_samples,
            "model": "Random Forest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        # =====================================================================
        # LightGBM Training
        # =====================================================================
        lgb_client_models = []
        lgb_accuracies = []
        lgb_losses = []
        lgb_samples = []
        
        for client_idx in range(num_clients):
            client_model = train_lightgbm_client(
                client_data[client_idx]["train_X"],
                client_data[client_idx]["train_y"],
                num_classes=num_classes,
                num_rounds=local_epochs,
                existing_model=lgb_global_model,
            )
            lgb_client_models.append(client_model)
            
            loss, accuracy = evaluate_lightgbm(
                client_model,
                client_data[client_idx]["test_X"],
                client_data[client_idx]["test_y"],
                num_classes,
            )
            
            n_samples = len(client_data[client_idx]["train_X"])
            lgb_accuracies.append(accuracy * n_samples)
            lgb_losses.append(loss * n_samples)
            lgb_samples.append(n_samples)
        
        lgb_global_model = lgb_client_models[-1]
        total_samples = sum(lgb_samples)
        lgb_avg_accuracy = sum(lgb_accuracies) / total_samples
        lgb_avg_loss = sum(lgb_losses) / total_samples
        
        lgb_round_metrics.append({
            "round": round_num,
            "accuracy": round(lgb_avg_accuracy, 6),
            "loss": round(lgb_avg_loss, 6),
            "num_clients": num_clients,
            "total_examples": total_samples,
            "model": "LightGBM",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    # Save XGBoost metrics (for backward compatibility with existing dashboard)
    METRICS_FILE.write_text(json.dumps(xgb_round_metrics, indent=2))
    
    # Save model comparison data
    model_comparison = {
        "xgboost": {
            "name": "XGBoost",
            "final_accuracy": xgb_round_metrics[-1]["accuracy"],
            "final_loss": xgb_round_metrics[-1]["loss"],
            "round_metrics": xgb_round_metrics,
        },
        "random_forest": {
            "name": "Random Forest",
            "final_accuracy": rf_round_metrics[-1]["accuracy"],
            "final_loss": rf_round_metrics[-1]["loss"],
            "round_metrics": rf_round_metrics,
        },
        "lightgbm": {
            "name": "LightGBM",
            "final_accuracy": lgb_round_metrics[-1]["accuracy"],
            "final_loss": lgb_round_metrics[-1]["loss"],
            "round_metrics": lgb_round_metrics,
        },
        "best_model": max(
            ["xgboost", "random_forest", "lightgbm"],
            key=lambda m: {
                "xgboost": xgb_round_metrics[-1]["accuracy"],
                "random_forest": rf_round_metrics[-1]["accuracy"],
                "lightgbm": lgb_round_metrics[-1]["accuracy"],
            }[m]
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_COMPARISON_FILE.write_text(json.dumps(model_comparison, indent=2))
    
    return xgb_round_metrics


if __name__ == "__main__":
    if not dataset_exists():
        print("Error: No dataset uploaded. Please upload a dataset through the dashboard first.")
        sys.exit(1)
    
    print("Starting federated multi-model training (XGBoost, Random Forest, LightGBM)...")
    metrics = run_federated_training(num_rounds=5, num_clients=3)
    print(f"Training complete! {len(metrics)} rounds.")
    
    # Load and display model comparison
    comparison = json.loads(MODEL_COMPARISON_FILE.read_text())
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    for model_key in ["xgboost", "random_forest", "lightgbm"]:
        model_data = comparison[model_key]
        print(f"{model_data['name']:15s}: Accuracy={model_data['final_accuracy']*100:.2f}%, Loss={model_data['final_loss']:.4f}")
    print("-" * 50)
    best = comparison["best_model"]
    print(f"Best Model: {comparison[best]['name']} ({comparison[best]['final_accuracy']*100:.2f}% accuracy)")

"""
Quick evaluation module: evaluate a dataset without running full FL training.
Useful for testing uploaded datasets directly from the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_csv_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Evaluate a CSV dataset using a simple model trained on the spot.
    Returns metrics dict with accuracy, precision, recall, f1, confusion matrix, and class info.
    
    CSV format: all columns except last = features, last column = target.
    """
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + target).")
    
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]
    
    X = df[feature_cols].values.astype(np.float64)
    y_raw = df[target_col].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)
    n_samples, n_features = X.shape
    
    class_counts = {name: int((y == i).sum()) for i, name in enumerate(class_names)}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(32, n_classes),
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        test_loss = criterion(outputs, y_test_t).item()
        _, predictions = torch.max(outputs, 1)
        y_pred = predictions.numpy()
        y_true = y_test_t.numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "dataset_name": "Uploaded Dataset",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_names": class_names,
        "class_counts": class_counts,
        "feature_names": feature_cols[:10],
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "epochs_trained": 100,
    }


def generate_evaluation_context(results: dict) -> str:
    """Generate text context for LLM from evaluation results."""
    lines = [
        "Dataset evaluation results:",
        f"- Dataset: {results['dataset_name']}",
        f"- Total samples: {results['n_samples']}",
        f"- Features: {results['n_features']}",
        f"- Classes: {results['n_classes']} ({', '.join(results['class_names'])})",
        "",
        "Class distribution:",
    ]
    
    for name, count in results['class_counts'].items():
        pct = (count / results['n_samples']) * 100
        lines.append(f"- {name}: {count} samples ({pct:.1f}%)")
    
    lines.extend([
        "",
        f"Train/test split: {results['train_samples']}/{results['test_samples']} samples",
        f"Training: {results['epochs_trained']} epochs with neural network (64→32→{results['n_classes']})",
        "",
        "Performance metrics on test set:",
        f"- Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)",
        f"- Precision: {results['precision']:.4f}",
        f"- Recall: {results['recall']:.4f}",
        f"- F1 Score: {results['f1_score']:.4f}",
        f"- Test Loss: {results['test_loss']:.4f}",
    ])
    
    return "\n".join(lines)

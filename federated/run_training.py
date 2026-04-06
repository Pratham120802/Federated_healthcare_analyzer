"""
Run complete federated training with improved model architecture.
Optimized for healthcare data classification with better accuracy.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated.data_loader import dataset_exists, get_dataset_summary, load_custom_csv, prepare_partitioned_data

METRICS_FILE = PROJECT_ROOT / "metrics.json"


class HealthcareModel(nn.Module):
    """
    Improved neural network for healthcare classification.
    Features: BatchNorm, residual-like connections, proper regularization.
    """
    def __init__(self, input_size: int, num_classes: int = 2):
        super().__init__()
        
        # Adaptive hidden sizes based on input
        h1 = min(256, max(64, input_size * 2))
        h2 = min(128, max(32, input_size))
        h3 = min(64, max(16, input_size // 2))
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.block1 = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Linear(h3, num_classes)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def prepare_client_data(num_clients: int = 3):
    """Prepare data partitions for all clients."""
    X_train_0, y_train_0, X_test, y_test, input_size = prepare_partitioned_data(0, num_clients)
    
    client_data = [{
        "train_X": torch.tensor(X_train_0, dtype=torch.float32),
        "train_y": torch.tensor(y_train_0, dtype=torch.long),
        "test_X": torch.tensor(X_test, dtype=torch.float32),
        "test_y": torch.tensor(y_test, dtype=torch.long),
    }]
    
    for i in range(1, num_clients):
        X_train_i, y_train_i, _, _, _ = prepare_partitioned_data(i, num_clients)
        client_data.append({
            "train_X": torch.tensor(X_train_i, dtype=torch.float32),
            "train_y": torch.tensor(y_train_i, dtype=torch.long),
            "test_X": torch.tensor(X_test, dtype=torch.float32),
            "test_y": torch.tensor(y_test, dtype=torch.long),
        })
    
    return client_data, input_size


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train_client(model, train_X, train_y, epochs=5, lr=0.001, batch_size=32):
    """
    Train model on client data with mini-batches and learning rate scheduling.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    
    return get_parameters(model), len(train_X)


def evaluate_client(model, test_X, test_y, batch_size=64):
    """Evaluate model on test data."""
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(test_X, test_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch_y).sum().item()
            total_samples += len(batch_y)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, total_samples


def federated_averaging(all_params, all_num_samples):
    """Average model parameters weighted by number of samples."""
    total_samples = sum(all_num_samples)
    
    avg_params = []
    for param_idx in range(len(all_params[0])):
        # Use float64 for accumulation to avoid casting issues
        weighted_sum = np.zeros_like(all_params[0][param_idx], dtype=np.float64)
        for client_idx, params in enumerate(all_params):
            weight = all_num_samples[client_idx] / total_samples
            weighted_sum += params[param_idx].astype(np.float64) * weight
        # Cast back to original dtype
        avg_params.append(weighted_sum.astype(all_params[0][param_idx].dtype))
    
    return avg_params


def run_federated_training(
    num_rounds: int = 5,
    num_clients: int = 3,
    local_epochs: int = 5,
    learning_rate: float = 0.001,
    batch_size: int = 32,
):
    """
    Run complete federated training simulation with improved settings.
    Returns list of round metrics.
    """
    if not dataset_exists():
        raise FileNotFoundError("No dataset uploaded. Please upload a dataset first.")
    
    ds_summary = get_dataset_summary()
    num_classes = ds_summary["n_classes"]
    
    # Adjust learning rate based on dataset size
    n_samples = ds_summary["n_samples"]
    if n_samples < 1000:
        learning_rate = 0.01  # Larger LR for small datasets
        local_epochs = 10  # More epochs
    elif n_samples > 10000:
        learning_rate = 0.0005  # Smaller LR for large datasets
        batch_size = 64
    
    client_data, input_size = prepare_client_data(num_clients=num_clients)
    
    global_model = HealthcareModel(input_size, num_classes)
    global_params = get_parameters(global_model)
    
    round_metrics = []
    best_accuracy = 0.0
    
    for round_num in range(1, num_rounds + 1):
        all_params = []
        all_num_samples = []
        all_accuracies = []
        all_losses = []
        all_eval_samples = []
        
        # Decay learning rate over rounds
        round_lr = learning_rate * (0.95 ** (round_num - 1))
        
        for client_idx in range(num_clients):
            client_model = HealthcareModel(input_size, num_classes)
            set_parameters(client_model, global_params)
            
            params, num_samples = train_client(
                client_model,
                client_data[client_idx]["train_X"],
                client_data[client_idx]["train_y"],
                epochs=local_epochs,
                lr=round_lr,
                batch_size=batch_size,
            )
            all_params.append(params)
            all_num_samples.append(num_samples)
            
            loss, accuracy, eval_samples = evaluate_client(
                client_model,
                client_data[client_idx]["test_X"],
                client_data[client_idx]["test_y"],
            )
            all_accuracies.append(accuracy * eval_samples)
            all_losses.append(loss * eval_samples)
            all_eval_samples.append(eval_samples)
        
        global_params = federated_averaging(all_params, all_num_samples)
        set_parameters(global_model, global_params)
        
        total_eval_samples = sum(all_eval_samples)
        avg_accuracy = sum(all_accuracies) / total_eval_samples
        avg_loss = sum(all_losses) / total_eval_samples
        
        best_accuracy = max(best_accuracy, avg_accuracy)
        
        round_metrics.append({
            "round": round_num,
            "accuracy": round(avg_accuracy, 6),
            "loss": round(avg_loss, 6),
            "num_clients": num_clients,
            "total_examples": sum(all_num_samples),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    METRICS_FILE.write_text(json.dumps(round_metrics, indent=2))
    
    return round_metrics


if __name__ == "__main__":
    if not dataset_exists():
        print("Error: No dataset uploaded. Please upload a dataset through the dashboard first.")
        sys.exit(1)
    
    print("Starting federated training...")
    metrics = run_federated_training(num_rounds=5, num_clients=3)
    print(f"Training complete! {len(metrics)} rounds.")
    for m in metrics:
        print(f"  Round {m['round']}: accuracy={m['accuracy']:.4f}, loss={m['loss']:.4f}")

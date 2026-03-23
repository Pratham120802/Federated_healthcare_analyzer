from typing import Tuple
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_partition(client_id: int, num_clients: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Partition training set across clients
    X_splits = np.array_split(X_train, num_clients)
    y_splits = np.array_split(y_train, num_clients)

    X_client = X_splits[client_id]
    y_client = y_splits[client_id]

    train_data = torch.tensor(X_client, dtype=torch.float32)
    train_labels = torch.tensor(y_client, dtype=torch.long)

    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    return train_data, train_labels, test_data, test_labels
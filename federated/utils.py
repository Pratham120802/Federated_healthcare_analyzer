"""
Utility functions for federated learning data preparation.
Uses user-uploaded dataset only (no default).
"""

from typing import Tuple

import torch

from federated.data_loader import prepare_partitioned_data


def load_partition(
    client_id: int,
    num_clients: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and partition data for a federated client.
    Uses the dataset uploaded by the user.
    Raises FileNotFoundError if no dataset uploaded.
    """
    X_train, y_train, X_test, y_test, _ = prepare_partitioned_data(
        client_id=client_id,
        num_clients=num_clients,
    )

    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    return train_data, train_labels, test_data, test_labels

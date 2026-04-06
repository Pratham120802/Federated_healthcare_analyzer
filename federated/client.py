import sys
from collections import OrderedDict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from model.model import HealthcareModel
from federated.data_loader import dataset_exists, get_dataset_summary
from federated.utils import load_partition


if len(sys.argv) < 2:
    raise ValueError("Usage: python federated/client.py <client_id>")

if not dataset_exists():
    raise FileNotFoundError(
        "No dataset found! Please upload a CSV dataset through the dashboard first.\n"
        "Run: python -m streamlit run dashboard/app.py"
    )

client_id = int(sys.argv[1])
num_clients = 3

train_data, train_labels, test_data, test_labels = load_partition(client_id, num_clients)
input_size = train_data.shape[1]

ds_summary = get_dataset_summary()
num_classes = ds_summary["n_classes"]

model = HealthcareModel(input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train():
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        loss = criterion(outputs, test_labels).item()
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):
        set_parameters(model, parameters)
        train()
        return get_parameters(model), len(train_data), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = test()
        return float(loss), len(test_data), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
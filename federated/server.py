import json
import flwr as fl


round_metrics = []


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    acc = sum(accuracies) / sum(examples)

    round_metrics.append({"accuracy": acc})

    with open("metrics.json", "w") as f:
        json.dump(round_metrics, f, indent=2)

    return {"accuracy": acc}


def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
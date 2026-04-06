import json
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl

METRICS_FILE = Path(__file__).resolve().parent.parent / "metrics.json"

round_metrics: list[dict] = []
_current_round = 0


def weighted_average(metrics):
    global _current_round
    _current_round += 1

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    acc = sum(accuracies) / total_examples
    loss = sum(losses) / total_examples if any(m.get("loss") for _, m in metrics) else 1.0 - acc

    round_metrics.append({
        "round": _current_round,
        "accuracy": round(acc, 6),
        "loss": round(loss, 6),
        "num_clients": len(metrics),
        "total_examples": total_examples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    METRICS_FILE.write_text(json.dumps(round_metrics, indent=2))
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

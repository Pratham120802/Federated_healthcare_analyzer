"""
Runtime description of the user's uploaded dataset for LLM context.
"""

from __future__ import annotations

from federated.data_loader import get_dataset_summary, dataset_exists


def get_training_data_context() -> str:
    """
    Generate a text summary of the uploaded dataset for LLM context.
    Returns empty string if no dataset uploaded.
    """
    if not dataset_exists():
        return ""
    
    summary = get_dataset_summary()

    lines = [
        "Dataset context (uploaded by user for federated training):",
        f"- Dataset file: {summary['name']}",
        f"- Total samples: {summary['n_samples']}",
        f"- Number of features: {summary['n_features']}",
        f"- Number of classes: {summary['n_classes']}",
    ]

    for class_name, count in summary["class_counts"].items():
        pct = (count / summary["n_samples"]) * 100
        lines.append(f"- Class '{class_name}': {count} samples ({pct:.1f}%)")

    if summary["feature_names"]:
        feat_preview = ", ".join(summary["feature_names"][:5])
        if len(summary["feature_names"]) > 5:
            feat_preview += ", ..."
        lines.append(f"- Features (sample): {feat_preview}")

    lines.append(
        "- Use these counts to answer questions about class distribution or sample sizes."
    )

    return "\n".join(lines)

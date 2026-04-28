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
        f"- Prediction target (disease/condition): {summary['target_column']}",
        f"- Total samples: {summary['n_samples']}",
        f"- Number of features: {summary['n_features']}",
        f"- Number of classes: {summary['n_classes']}",
    ]

    # Add class distribution with clear labels
    lines.append("- Outcome classes:")
    for class_name, count in summary["class_counts"].items():
        pct = (count / summary["n_samples"]) * 100
        lines.append(f"  * '{class_name}': {count} samples ({pct:.1f}%)")

    if summary.get("health_features"):
        health_preview = ", ".join(summary["health_features"][:5])
        if len(summary["health_features"]) > 5:
            health_preview += ", ..."
        lines.append(f"- Health-related features: {health_preview}")

    if summary["feature_names"]:
        feat_preview = ", ".join(summary["feature_names"][:5])
        if len(summary["feature_names"]) > 5:
            feat_preview += ", ..."
        lines.append(f"- All features (sample): {feat_preview}")

    return "\n".join(lines)


def get_dataset_info_for_display() -> dict | None:
    """
    Get dataset information formatted for UI display.
    Returns None if no dataset uploaded.
    """
    if not dataset_exists():
        return None
    
    summary = get_dataset_summary()
    
    return {
        "dataset_name": summary["name"],
        "target_column": summary["target_column"],
        "n_samples": summary["n_samples"],
        "n_features": summary["n_features"],
        "n_classes": summary["n_classes"],
        "class_names": summary["class_names"],
        "class_counts": summary["class_counts"],
        "feature_names": summary["feature_names"],
        "health_features": summary.get("health_features", []),
    }

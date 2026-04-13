"""
Data loading with smart healthcare feature detection and improved preprocessing.
Automatically identifies disease/health-related columns and optimizes for better accuracy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DATA_FILE = PROJECT_ROOT / "custom_dataset.csv"
DATA_CONFIG_FILE = PROJECT_ROOT / "data_config.json"

SUPPORTED_FORMATS = {
    "csv": "CSV (Comma Separated)",
    "xlsx": "Excel (.xlsx)",
    "xls": "Excel (.xls)",
    "json": "JSON",
    "parquet": "Parquet",
    "tsv": "TSV (Tab Separated)",
    "txt": "Text (Tab/Comma)",
}

# Healthcare/disease related keywords for smart column detection
HEALTH_KEYWORDS = [
    # Disease terms
    'disease', 'diagnosis', 'patient', 'clinical', 'symptom', 'treatment',
    'malignant', 'benign', 'tumor', 'cancer', 'infection', 'positive', 'negative',
    # Medical measurements
    'blood', 'pressure', 'glucose', 'cholesterol', 'bmi', 'heart', 'rate',
    'temperature', 'oxygen', 'saturation', 'hemoglobin', 'platelet',
    # Test results
    'test', 'result', 'score', 'level', 'count', 'value', 'measure',
    'rdt', 'pcr', 'tpr', 'fnr', 'sensitivity', 'specificity',
    # Health indicators
    'age', 'weight', 'height', 'gender', 'sex', 'smoker', 'diabetic',
    'hypertension', 'pregnant', 'mortality', 'survival', 'risk',
    # Medical categories
    'stage', 'grade', 'class', 'type', 'category', 'status', 'outcome',
    'severity', 'prognosis', 'response', 'recurrence',
]

# Columns to exclude (usually not useful for prediction)
EXCLUDE_KEYWORDS = [
    'id', 'name', 'date', 'time', 'timestamp', 'index', 'uuid',
    'phone', 'email', 'address', 'comment', 'note', 'description',
]


def dataset_exists() -> bool:
    """Check if user has uploaded a dataset."""
    return CUSTOM_DATA_FILE.exists()


def get_data_config() -> dict:
    """Get data configuration (target column, etc.)."""
    if DATA_CONFIG_FILE.exists():
        return json.loads(DATA_CONFIG_FILE.read_text())
    return {}


def set_data_config(target_column: str | None = None, original_filename: str | None = None):
    """Set data configuration."""
    config = get_data_config()
    if target_column:
        config["target_column"] = target_column
    if original_filename:
        config["original_filename"] = original_filename
    DATA_CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_csv_columns() -> list[str]:
    """Get column names from uploaded dataset."""
    if not CUSTOM_DATA_FILE.exists():
        return []
    df = pd.read_csv(CUSTOM_DATA_FILE, nrows=0)
    return df.columns.tolist()


def get_column_info() -> list[dict]:
    """Get info about each column (name, dtype, sample values, null count)."""
    if not CUSTOM_DATA_FILE.exists():
        return []
    df = pd.read_csv(CUSTOM_DATA_FILE, nrows=1000)
    info = []
    for col in df.columns:
        n_unique = df[col].nunique()
        n_null = df[col].isna().sum()
        sample = df[col].dropna().head(3).tolist()
        info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "n_null": n_null,
            "sample": sample,
        })
    return info


def _is_health_related(col_name: str) -> bool:
    """Check if column name is health/disease related."""
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in HEALTH_KEYWORDS)


def _should_exclude(col_name: str) -> bool:
    """Check if column should be excluded."""
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in EXCLUDE_KEYWORDS)


def _get_feature_importance_score(col_name: str) -> int:
    """Score column by healthcare relevance (higher = more relevant)."""
    col_lower = col_name.lower()
    score = 0
    
    # Boost health-related columns
    for kw in HEALTH_KEYWORDS:
        if kw in col_lower:
            score += 10
    
    # Penalize excluded columns
    for kw in EXCLUDE_KEYWORDS:
        if kw in col_lower:
            score -= 50
    
    return score


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded file into a DataFrame."""
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif filename.endswith(".json"):
        try:
            df = pd.read_json(uploaded_file)
        except ValueError:
            uploaded_file.seek(0)
            df = pd.read_json(uploaded_file, lines=True)
        return df
    elif filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif filename.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    elif filename.endswith(".txt"):
        uploaded_file.seek(0)
        first_line = uploaded_file.readline().decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        if "\t" in first_line:
            return pd.read_csv(uploaded_file, sep="\t")
        else:
            return pd.read_csv(uploaded_file)
    else:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            raise ValueError(f"Unsupported file format: {filename}")


def _smart_feature_selection(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> list[str]:
    """
    Smart feature selection prioritizing health-related columns.
    Returns optimized list of feature columns.
    """
    # Score all columns
    col_scores = {col: _get_feature_importance_score(col) for col in feature_cols}
    
    # Separate into health-related and other
    health_cols = [c for c in feature_cols if col_scores[c] > 0]
    other_cols = [c for c in feature_cols if col_scores[c] <= 0 and col_scores[c] > -50]
    excluded_cols = [c for c in feature_cols if col_scores[c] <= -50]
    
    # Prioritize health columns, then others (exclude ID-like columns)
    selected = health_cols + other_cols
    
    # If we have too many features, limit to most relevant
    max_features = min(50, len(selected))
    if len(selected) > max_features:
        # Sort by score and take top
        selected = sorted(selected, key=lambda c: col_scores[c], reverse=True)[:max_features]
    
    return selected if selected else feature_cols


def _preprocess_features(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    """
    Enhanced preprocessing:
    - Drop useless columns (all NaN, single value, ID-like)
    - Encode categorical columns
    - Handle missing values intelligently
    - Returns processed DataFrame and final column names
    """
    X_df = df[feature_cols].copy()
    
    # Drop columns that are entirely NaN
    X_df = X_df.dropna(axis=1, how='all')
    
    # Drop columns with single unique value (no information)
    cols_to_drop = []
    for col in X_df.columns.tolist():
        if X_df[col].nunique() <= 1:
            cols_to_drop.append(col)
    X_df = X_df.drop(columns=cols_to_drop)
    
    # Drop columns with too many unique values relative to rows (likely IDs)
    n_rows = len(X_df)
    cols_to_drop = []
    for col in X_df.columns.tolist():
        if X_df[col].dtype == 'object':
            unique_ratio = X_df[col].nunique() / n_rows
            if unique_ratio > 0.9:  # 90% unique = probably an ID
                cols_to_drop.append(col)
    X_df = X_df.drop(columns=cols_to_drop)
    
    # Process remaining columns - encode ALL to numeric
    for col in X_df.columns.tolist():
        # Check if column is actually numeric
        is_numeric = pd.api.types.is_numeric_dtype(X_df[col])
        
        if not is_numeric or X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
            # Categorical/string: fill NaN and encode
            X_df[col] = X_df[col].fillna('_MISSING_').astype(str)
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])
        else:
            # Numeric: fill NaN with median
            if X_df[col].isna().any():
                median_val = X_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_df[col] = X_df[col].fillna(median_val)
    
    # Ensure all columns are numeric
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
    
    return X_df, X_df.columns.tolist()


def load_custom_data() -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load user dataset with smart feature selection.
    Returns (X, y, feature_names, class_names).
    """
    if not CUSTOM_DATA_FILE.exists():
        raise FileNotFoundError(
            "No dataset found. Please upload a file through the dashboard first."
        )

    config = get_data_config()
    df = pd.read_csv(CUSTOM_DATA_FILE)
    
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns.")

    # Determine target column
    target_col = config.get("target_column")
    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]
    
    # Feature columns = all except target
    all_feature_cols = [c for c in df.columns if c != target_col]
    
    # Smart feature selection
    feature_cols = _smart_feature_selection(df, all_feature_cols, target_col)
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    
    if len(df) == 0:
        raise ValueError(f"Target column '{target_col}' has no valid values.")

    # Preprocess features
    X_df, feature_names = _preprocess_features(df, feature_cols)
    
    if X_df.shape[1] == 0:
        raise ValueError("No valid features found after preprocessing.")
    
    X = X_df.values.astype(np.float64)

    # Encode target
    y_raw = df[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str)).astype(np.int64)
    class_names = [str(c) for c in le.classes_]

    return X, y, feature_names, class_names


# Alias for backward compatibility
load_custom_csv = load_custom_data


def save_custom_dataset(df: pd.DataFrame, original_filename: str = "dataset"):
    """Save uploaded dataframe to project root as CSV."""
    df.to_csv(CUSTOM_DATA_FILE, index=False)
    config = {"original_filename": original_filename}
    DATA_CONFIG_FILE.write_text(json.dumps(config, indent=2))


def delete_custom_dataset():
    """Remove uploaded dataset and config."""
    if CUSTOM_DATA_FILE.exists():
        CUSTOM_DATA_FILE.unlink()
    if DATA_CONFIG_FILE.exists():
        DATA_CONFIG_FILE.unlink()


def get_dataset_summary() -> dict:
    """Get summary statistics for LLM context."""
    X, y, feature_names, class_names = load_custom_data()
    n_samples, n_features = X.shape
    n_classes = len(class_names)

    class_counts = {}
    for i, name in enumerate(class_names):
        class_counts[name] = int((y == i).sum())

    config = get_data_config()
    target_col = config.get("target_column", "last column")
    original_name = config.get("original_filename", "custom_dataset")

    # Identify health-related features
    health_features = [f for f in feature_names if _is_health_related(f)]

    return {
        "name": original_name,
        "target_column": target_col,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_names": class_names,
        "class_counts": class_counts,
        "feature_names": feature_names[:10],
        "health_features": health_features[:10],
    }


def prepare_partitioned_data(
    client_id: int,
    num_clients: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load, split, normalize, and partition data for federated learning.
    Uses stratified splitting and robust normalization.
    """
    X, y, _, class_names = load_custom_data()
    n_classes = len(class_names)
    
    # Check if stratification is possible
    class_counts = np.bincount(y)
    min_class_count = class_counts.min()
    
    if min_class_count < 2:
        # Can't stratify - use random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Stratified split for balanced classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    # Robust normalization using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Handle any remaining NaN/inf from scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Partition training data across clients
    X_splits = np.array_split(X_train, num_clients)
    y_splits = np.array_split(y_train, num_clients)

    return (
        X_splits[client_id],
        y_splits[client_id],
        X_test,
        y_test,
        X.shape[1],
    )

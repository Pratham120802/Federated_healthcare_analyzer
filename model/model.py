"""
XGBoost model for healthcare classification.
Used in federated learning setup.
"""

import xgboost as xgb
import numpy as np


def create_xgboost_model(num_classes: int = 2) -> dict:
    """
    Create XGBoost parameters for healthcare classification.
    
    Args:
        num_classes: Number of target classes
        
    Returns:
        Dictionary of XGBoost parameters
    """
    params = {
        'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
        'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'seed': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }
    
    if num_classes > 2:
        params['num_class'] = num_classes
    
    return params


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    num_rounds: int = 100,
    early_stopping_rounds: int = 10,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> xgb.Booster:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_classes: Number of target classes
        num_rounds: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Trained XGBoost Booster
    """
    params = create_xgboost_model(num_classes)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    evals = [(dtrain, 'train')]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'val'))
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
        verbose_eval=False,
    )
    
    return model


def predict(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with XGBoost model.
    
    Args:
        model: Trained XGBoost Booster
        X: Features to predict
        
    Returns:
        Predicted class labels
    """
    dtest = xgb.DMatrix(X)
    probs = model.predict(dtest)
    
    if len(probs.shape) > 1:
        # Multi-class: return argmax
        return np.argmax(probs, axis=1)
    else:
        # Binary: threshold at 0.5
        return (probs > 0.5).astype(int)


def predict_proba(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Get prediction probabilities.
    
    Args:
        model: Trained XGBoost Booster
        X: Features to predict
        
    Returns:
        Prediction probabilities
    """
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)

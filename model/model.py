"""
Machine learning models for healthcare classification.
Supports XGBoost, Random Forest, and LightGBM.
Used in federated learning setup.
"""

import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# =============================================================================
# XGBoost Model
# =============================================================================

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


# =============================================================================
# Random Forest Model
# =============================================================================

def create_random_forest_model(num_classes: int = 2) -> RandomForestClassifier:
    """
    Create Random Forest classifier for healthcare classification.
    
    Args:
        num_classes: Number of target classes
        
    Returns:
        Configured RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' if num_classes > 2 else None,
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> RandomForestClassifier:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_classes: Number of target classes
        X_val: Validation features (optional, not used but kept for API consistency)
        y_val: Validation labels (optional, not used but kept for API consistency)
        
    Returns:
        Trained RandomForestClassifier
    """
    model = create_random_forest_model(num_classes)
    model.fit(X_train, y_train)
    return model


def predict_rf(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with Random Forest model.
    
    Args:
        model: Trained RandomForestClassifier
        X: Features to predict
        
    Returns:
        Predicted class labels
    """
    return model.predict(X)


def predict_proba_rf(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Get prediction probabilities from Random Forest.
    
    Args:
        model: Trained RandomForestClassifier
        X: Features to predict
        
    Returns:
        Prediction probabilities
    """
    return model.predict_proba(X)


# =============================================================================
# LightGBM Model
# =============================================================================

def create_lightgbm_params(num_classes: int = 2) -> dict:
    """
    Create LightGBM parameters for healthcare classification.
    
    Args:
        num_classes: Number of target classes
        
    Returns:
        Dictionary of LightGBM parameters
    """
    params = {
        'objective': 'multiclass' if num_classes > 2 else 'binary',
        'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    if num_classes > 2:
        params['num_class'] = num_classes
    
    return params


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    num_rounds: int = 100,
    early_stopping_rounds: int = 10,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> lgb.Booster:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_classes: Number of target classes
        num_rounds: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Trained LightGBM Booster
    """
    params = create_lightgbm_params(num_classes)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    valid_names = ['train']
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append('val')
    
    callbacks = []
    if X_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
    callbacks.append(lgb.log_evaluation(period=0))
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    
    return model


def predict_lgb(model: lgb.Booster, X: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Make predictions with LightGBM model.
    
    Args:
        model: Trained LightGBM Booster
        X: Features to predict
        num_classes: Number of classes
        
    Returns:
        Predicted class labels
    """
    probs = model.predict(X)
    
    if num_classes > 2:
        return np.argmax(probs, axis=1)
    else:
        return (probs > 0.5).astype(int)


def predict_proba_lgb(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Get prediction probabilities from LightGBM.
    
    Args:
        model: Trained LightGBM Booster
        X: Features to predict
        
    Returns:
        Prediction probabilities
    """
    return model.predict(X)

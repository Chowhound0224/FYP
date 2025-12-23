"""Model training with Optuna hyperparameter optimization (XGBoost only)."""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from typing import Tuple
import xgboost as xgb

from src.config import OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_N_JOBS, RANDOM_STATE


def train_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple:
    """
    Train XGBoost model using Optuna hyperparameter optimization.

    Optimizes XGBoost hyperparameters using TPE sampler:
        - n_estimators: Number of boosting rounds
        - max_depth: Maximum tree depth
        - learning_rate: Step size shrinkage
        - subsample: Row sampling ratio
        - colsample_bytree: Column sampling ratio
        - min_child_weight: Minimum sum of instance weight
        - gamma: Minimum loss reduction for split
        - reg_alpha: L1 regularization
        - reg_lambda: L2 regularization
    """

    print("\nStarting Optuna hyperparameter optimization for XGBoost...")
    print(f"Trying {OPTUNA_N_TRIALS} configurations...")

    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight('balanced', y=y_train)
    print(f"[OK] Computed sample weights for {len(np.unique(y_train))} classes")

    def objective(trial):
        # XGBoost hyperparameter search space
        clf = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            gamma=trial.suggest_float('gamma', 0.0, 5.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
            tree_method="hist",
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )

        # Cross-validation
        scores = cross_val_score(
            clf,
            X_train,
            y_train,
            cv=OPTUNA_CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=OPTUNA_N_JOBS
        )

        return scores.mean()

    # ------------------------------------------------------------------
    # Run Optuna Optimization
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n[OK] Best F1 Score: {study.best_value:.4f}")
    print(f"[OK] Best XGBoost Parameters: {best_params}")

    # ------------------------------------------------------------------
    # Train the final XGBoost model with best parameters
    # ------------------------------------------------------------------
    best_model = xgb.XGBClassifier(
        **best_params,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )

    best_model.fit(X_train, y_train, sample_weight=sample_weights)

    return best_model, best_params, study.best_value




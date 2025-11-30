"""Model training with Optuna hyperparameter optimization (XGBoost, LightGBM, Logistic, RandomForest)."""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Tuple
import xgboost as xgb
import lightgbm as lgb

from src.config import OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_N_JOBS, RANDOM_STATE


def train_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple:
    """
    Train model using Optuna hyperparameter optimization.

    Tries 4 model types:
        - XGBoost
        - LightGBM
        - Logistic Regression
        - Random Forest

    Optimizes hyperparameters using TPE sampler.
    """

    print("\nStarting Optuna hyperparameter optimization...")
    print(f"Trying {OPTUNA_N_TRIALS} configurations...")

    def objective(trial):
        model_type = trial.suggest_categorical(
            'model',
            ['xgboost', 'lightgbm', 'logistic', 'random_forest']
        )

        # ---------------------------------------------------------
        # XGBOOST
        # ---------------------------------------------------------
        if model_type == 'xgboost':
            clf = xgb.XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 150, 500),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                gamma=trial.suggest_float('gamma', 0.0, 5.0),
                tree_method="hist",
                random_state=RANDOM_STATE,
                eval_metric='logloss'
            )

        # ---------------------------------------------------------
        # LIGHTGBM
        # ---------------------------------------------------------
        elif model_type == 'lightgbm':
            clf = lgb.LGBMClassifier(
                num_leaves=trial.suggest_int('num_leaves', 20, 150),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                n_estimators=trial.suggest_int('n_estimators', 150, 500),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        # ---------------------------------------------------------
        # LOGISTIC REGRESSION
        # ---------------------------------------------------------
        elif model_type == 'logistic':
            clf = LogisticRegression(
                C=trial.suggest_float('C', 0.1, 10.0),
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )

        # ---------------------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------------------
        else:
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 500),
                max_depth=trial.suggest_int('max_depth', 10, 60),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        # Do cross-validation
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
    print(f"[OK] Best Parameters: {best_params}")

    # ------------------------------------------------------------------
    # Train the final model with best parameters
    # ------------------------------------------------------------------
    model_type = best_params['model']

    if model_type == 'xgboost':
        best_model = xgb.XGBClassifier(
            **{k: v for k, v in best_params.items() if k != 'model'},
            tree_method="hist",
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )

    elif model_type == 'lightgbm':
        best_model = lgb.LGBMClassifier(
            **{k: v for k, v in best_params.items() if k != 'model'},
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_type == 'logistic':
        best_model = LogisticRegression(
            C=best_params['C'],
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    else:
        best_model = RandomForestClassifier(
            **{k: v for k, v in best_params.items() if k != 'model'},
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    best_model.fit(X_train, y_train)

    return best_model, best_params, study.best_value




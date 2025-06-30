"""
Model Ensemble System for JARVIS
=================================

Advanced ensemble learning system for combining multiple models to improve
performance, robustness, and generalization.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import joblib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.optimize import minimize
from scipy.stats import mode
import optuna
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge, Summary
import hashlib
import tempfile
import shutil

logger = get_logger(__name__)

# Metrics
ensemble_created = Counter(
    "ensemble_models_created_total", "Total ensemble models created"
)
ensemble_predictions = Counter(
    "ensemble_predictions_total", "Total ensemble predictions"
)
ensemble_accuracy = Gauge("ensemble_accuracy", "Ensemble model accuracy", ["model_id"])
ensemble_training_time = Histogram(
    "ensemble_training_duration_seconds", "Ensemble training time"
)


@dataclass
class ModelMetadata:
    """Metadata for individual models in ensemble"""

    model_id: str
    model_type: str
    training_accuracy: float
    validation_accuracy: float
    weight: float = 1.0
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models"""

    ensemble_type: str  # voting, stacking, boosting, blending, dynamic
    task_type: str  # classification, regression
    combination_method: str  # average, weighted, meta-learner
    cv_folds: int = 5
    optimization_metric: str = "accuracy"  # accuracy, f1, mse, r2
    diversity_threshold: float = 0.3
    min_model_performance: float = 0.5
    max_models: int = 20
    use_feature_selection: bool = True
    calibration_method: Optional[str] = None  # isotonic, sigmoid
    dynamic_weighting: bool = False


@dataclass
class PredictionResult:
    """Result of ensemble prediction"""

    prediction: Any
    confidence: float
    model_contributions: Dict[str, float]
    uncertainty: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble"""

    overall_score: float
    individual_scores: Dict[str, float]
    diversity_score: float
    improvement_over_best: float
    metrics: Dict[str, Any]


class DynamicWeightOptimizer:
    """Optimizes ensemble weights dynamically"""

    def __init__(self):
        self.weight_history: List[np.ndarray] = []
        self.performance_history: List[float] = []

    def optimize_weights(
        self, predictions: np.ndarray, true_labels: np.ndarray, metric_fn: Callable
    ) -> np.ndarray:
        """Optimize weights using scipy optimization"""
        n_models = predictions.shape[1]

        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            ensemble_pred = np.average(predictions, weights=weights, axis=1)
            return -metric_fn(true_labels, ensemble_pred)

        # Initial weights (equal)
        initial_weights = np.ones(n_models) / n_models

        # Constraints: weights sum to 1, all weights >= 0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimized_weights = result.x / np.sum(result.x)
        self.weight_history.append(optimized_weights)
        self.performance_history.append(-result.fun)

        return optimized_weights


class BayesianModelAveraging:
    """Implements Bayesian Model Averaging for ensemble"""

    def __init__(self):
        self.posterior_weights: Optional[np.ndarray] = None
        self.evidence: Optional[np.ndarray] = None

    def compute_weights(
        self,
        predictions: List[np.ndarray],
        true_labels: np.ndarray,
        prior_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Bayesian posterior weights"""
        n_models = len(predictions)

        if prior_weights is None:
            prior_weights = np.ones(n_models) / n_models

        # Compute likelihood for each model
        likelihoods = []
        for pred in predictions:
            # Gaussian likelihood assumption
            residuals = true_labels - pred
            sigma = np.std(residuals)
            likelihood = np.exp(-0.5 * np.sum(residuals**2) / sigma**2)
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)

        # Compute evidence (marginal likelihood)
        self.evidence = np.sum(prior_weights * likelihoods)

        # Compute posterior weights
        self.posterior_weights = (prior_weights * likelihoods) / self.evidence

        return self.posterior_weights


class StackedGeneralization:
    """Implements stacked generalization (meta-learning)"""

    def __init__(self, meta_model: Optional[Any] = None):
        self.meta_model = meta_model or LogisticRegression()
        self.base_models: List[Any] = []
        self.is_fitted: bool = False

    def create_meta_features(
        self, X: np.ndarray, base_predictions: List[np.ndarray]
    ) -> np.ndarray:
        """Create meta-features from base model predictions"""
        # Stack predictions
        meta_features = np.column_stack(base_predictions)

        # Add diversity features
        if len(base_predictions) > 1:
            # Variance across predictions
            variance = np.var(meta_features, axis=1, keepdims=True)
            # Agreement score
            agreement = np.mean(
                [
                    np.equal(base_predictions[i], base_predictions[j])
                    for i in range(len(base_predictions))
                    for j in range(i + 1, len(base_predictions))
                ],
                axis=0,
            ).reshape(-1, 1)

            meta_features = np.hstack([meta_features, variance, agreement])

        return meta_features

    def fit(self, X: np.ndarray, y: np.ndarray, base_models: List[Any]):
        """Fit the meta-model"""
        self.base_models = base_models

        # Generate out-of-fold predictions
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = []
        meta_targets = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            val_predictions = []
            for model in base_models:
                # Clone and fit model
                cloned_model = type(model)(**model.get_params())
                cloned_model.fit(X_train, y_train)
                val_pred = cloned_model.predict(X_val)
                val_predictions.append(val_pred)

            meta_feat = self.create_meta_features(X_val, val_predictions)
            meta_features.append(meta_feat)
            meta_targets.append(y_val)

        # Train meta-model
        meta_features = np.vstack(meta_features)
        meta_targets = np.hstack(meta_targets)
        self.meta_model.fit(meta_features, meta_targets)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the stacked model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        base_predictions = [model.predict(X) for model in self.base_models]
        meta_features = self.create_meta_features(X, base_predictions)

        return self.meta_model.predict(meta_features)


class ModelEnsemble:
    """
    Advanced model ensemble system with multiple combination strategies.

    Features:
    - Multiple ensemble techniques (voting, stacking, boosting, blending)
    - Dynamic weight optimization
    - Bayesian model averaging
    - Model diversity management
    - Uncertainty quantification
    - AutoML integration
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.ensemble_model: Optional[Any] = None
        self.weight_optimizer = DynamicWeightOptimizer()
        self.bayesian_averager = BayesianModelAveraging()
        self.stacked_model = StackedGeneralization()
        self.feature_importance: Dict[str, float] = {}
        self.calibrators: Dict[str, Any] = {}
        self.performance_history: List[EnsemblePerformance] = []

        # Initialize storage
        self.storage_path = Path("model_ensembles")
        self.storage_path.mkdir(exist_ok=True)

        logger.info("Initialized ModelEnsemble", config=config)

    def add_model(self, model: Any, model_id: str, metadata: ModelMetadata) -> None:
        """Add a model to the ensemble"""
        if len(self.models) >= self.config.max_models:
            # Remove worst performing model
            worst_model = min(
                self.model_metadata.items(), key=lambda x: x[1].validation_accuracy
            )[0]
            del self.models[worst_model]
            del self.model_metadata[worst_model]

        self.models[model_id] = model
        self.model_metadata[model_id] = metadata

        logger.info(
            "Added model to ensemble",
            model_id=model_id,
            accuracy=metadata.validation_accuracy,
        )

    def compute_model_diversity(self, predictions: Dict[str, np.ndarray]) -> float:
        """Compute diversity score for model predictions"""
        if len(predictions) < 2:
            return 0.0

        pred_matrix = np.array(list(predictions.values()))

        # Compute pairwise disagreement
        disagreements = []
        for i in range(len(pred_matrix)):
            for j in range(i + 1, len(pred_matrix)):
                disagreement = np.mean(pred_matrix[i] != pred_matrix[j])
                disagreements.append(disagreement)

        return np.mean(disagreements)

    def select_diverse_models(
        self, candidate_models: Dict[str, Any], n_models: int
    ) -> List[str]:
        """Select diverse models for ensemble"""
        if len(candidate_models) <= n_models:
            return list(candidate_models.keys())

        selected = []
        remaining = list(candidate_models.keys())

        # Start with best performing model
        best_model = max(
            remaining, key=lambda x: self.model_metadata[x].validation_accuracy
        )
        selected.append(best_model)
        remaining.remove(best_model)

        # Greedily add models that maximize diversity
        while len(selected) < n_models and remaining:
            max_diversity = -1
            best_candidate = None

            for candidate in remaining:
                # Compute diversity with current selection
                temp_selected = selected + [candidate]
                predictions = {
                    m: self.models[m].predict(self.validation_data[0])
                    for m in temp_selected
                }
                diversity = self.compute_model_diversity(predictions)

                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def create_voting_ensemble(self) -> Any:
        """Create a voting ensemble"""
        estimators = [(model_id, model) for model_id, model in self.models.items()]

        if self.config.task_type == "classification":
            voting = (
                "soft"
                if hasattr(list(self.models.values())[0], "predict_proba")
                else "hard"
            )
            return VotingClassifier(estimators=estimators, voting=voting)
        else:
            weights = [self.model_metadata[m_id].weight for m_id in self.models.keys()]
            return VotingRegressor(estimators=estimators, weights=weights)

    def create_stacking_ensemble(self) -> Any:
        """Create a stacking ensemble"""
        base_estimators = [(model_id, model) for model_id, model in self.models.items()]

        # Choose meta-learner based on task
        if self.config.task_type == "classification":
            meta_learner = LogisticRegression()
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
            )
        else:
            meta_learner = Ridge()
            return StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
            )

    def create_boosting_ensemble(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Create a boosting ensemble"""
        if self.config.task_type == "classification":
            # XGBoost ensemble
            ensemble = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            # LightGBM ensemble
            ensemble = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
            )

        ensemble.fit(X, y)
        return ensemble

    def create_blending_ensemble(
        self, X_blend: np.ndarray, y_blend: np.ndarray
    ) -> Callable:
        """Create a blending ensemble"""
        # Get predictions from all models on blend set
        blend_predictions = []
        for model_id, model in self.models.items():
            pred = model.predict(X_blend)
            blend_predictions.append(pred)

        blend_features = np.column_stack(blend_predictions)

        # Train blender
        if self.config.task_type == "classification":
            blender = LogisticRegression()
        else:
            blender = Ridge()

        blender.fit(blend_features, y_blend)

        def blend_predict(X):
            predictions = []
            for model_id, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)

            features = np.column_stack(predictions)
            return blender.predict(features)

        return blend_predict

    async def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> EnsemblePerformance:
        """Train the ensemble model"""
        start_time = time.time()

        # Store validation data for model selection
        if X_val is not None:
            self.validation_data = (X_val, y_val)

        # Select diverse models if needed
        if self.config.use_feature_selection:
            selected_models = self.select_diverse_models(
                self.models, min(len(self.models), 10)
            )
            active_models = {k: self.models[k] for k in selected_models}
        else:
            active_models = self.models

        # Create ensemble based on type
        if self.config.ensemble_type == "voting":
            self.ensemble_model = self.create_voting_ensemble()
            self.ensemble_model.fit(X_train, y_train)

        elif self.config.ensemble_type == "stacking":
            self.ensemble_model = self.create_stacking_ensemble()
            self.ensemble_model.fit(X_train, y_train)

        elif self.config.ensemble_type == "boosting":
            self.ensemble_model = self.create_boosting_ensemble(X_train, y_train)

        elif self.config.ensemble_type == "blending":
            # Split training data for blending
            blend_size = int(0.2 * len(X_train))
            X_blend, y_blend = X_train[:blend_size], y_train[:blend_size]
            X_train_new = X_train[blend_size:]
            y_train_new = y_train[blend_size:]

            # Train base models on new training set
            for model in active_models.values():
                model.fit(X_train_new, y_train_new)

            self.ensemble_model = self.create_blending_ensemble(X_blend, y_blend)

        elif self.config.ensemble_type == "dynamic":
            # Train with dynamic weighting
            await self._train_dynamic_ensemble(X_train, y_train, X_val, y_val)

        # Evaluate performance
        performance = await self._evaluate_ensemble(X_val, y_val)
        self.performance_history.append(performance)

        # Update metrics
        training_time = time.time() - start_time
        ensemble_training_time.observe(training_time)
        ensemble_created.inc()

        logger.info(
            "Ensemble training completed",
            ensemble_type=self.config.ensemble_type,
            performance=performance.overall_score,
            training_time=training_time,
        )

        return performance

    async def _train_dynamic_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        """Train ensemble with dynamic weighting"""
        # Get predictions from all models
        train_predictions = []
        val_predictions = []

        for model_id, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            train_pred = model.predict(X_train)
            train_predictions.append(train_pred)

            if X_val is not None:
                val_pred = model.predict(X_val)
                val_predictions.append(val_pred)

        train_predictions = np.array(train_predictions).T

        # Optimize weights
        if self.config.task_type == "classification":
            metric_fn = lambda y_true, y_pred: accuracy_score(
                y_true, (y_pred > 0.5).astype(int)
            )
        else:
            metric_fn = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

        optimal_weights = self.weight_optimizer.optimize_weights(
            train_predictions, y_train, metric_fn
        )

        # Update model weights
        for i, model_id in enumerate(self.models.keys()):
            self.model_metadata[model_id].weight = optimal_weights[i]

        # Create weighted ensemble function
        def dynamic_predict(X):
            predictions = []
            weights = []

            for model_id, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.model_metadata[model_id].weight)

            predictions = np.array(predictions).T
            weights = np.array(weights)

            return np.average(predictions, weights=weights, axis=1)

        self.ensemble_model = dynamic_predict

    async def _evaluate_ensemble(
        self, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]
    ) -> EnsemblePerformance:
        """Evaluate ensemble performance"""
        if X_val is None or y_val is None:
            return EnsemblePerformance(
                overall_score=0.0,
                individual_scores={},
                diversity_score=0.0,
                improvement_over_best=0.0,
                metrics={},
            )

        # Get predictions
        if callable(self.ensemble_model):
            ensemble_pred = self.ensemble_model(X_val)
        else:
            ensemble_pred = self.ensemble_model.predict(X_val)

        # Calculate overall score
        if self.config.task_type == "classification":
            overall_score = accuracy_score(y_val, ensemble_pred)
            f1 = f1_score(y_val, ensemble_pred, average="weighted")
            metrics = {"accuracy": overall_score, "f1_score": f1}
        else:
            overall_score = r2_score(y_val, ensemble_pred)
            mse = mean_squared_error(y_val, ensemble_pred)
            metrics = {"r2_score": overall_score, "mse": mse}

        # Calculate individual model scores
        individual_scores = {}
        individual_predictions = {}

        for model_id, model in self.models.items():
            pred = model.predict(X_val)
            individual_predictions[model_id] = pred

            if self.config.task_type == "classification":
                score = accuracy_score(y_val, pred)
            else:
                score = r2_score(y_val, pred)

            individual_scores[model_id] = score

        # Calculate diversity
        diversity_score = self.compute_model_diversity(individual_predictions)

        # Calculate improvement
        best_individual = max(individual_scores.values())
        improvement = (overall_score - best_individual) / best_individual * 100

        # Update ensemble accuracy metric
        ensemble_accuracy.labels(model_id="ensemble").set(overall_score)

        return EnsemblePerformance(
            overall_score=overall_score,
            individual_scores=individual_scores,
            diversity_score=diversity_score,
            improvement_over_best=improvement,
            metrics=metrics,
        )

    def predict(
        self, X: np.ndarray, return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with the ensemble"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble must be trained before prediction")

        # Increment prediction counter
        ensemble_predictions.inc()

        # Get predictions
        if callable(self.ensemble_model):
            predictions = self.ensemble_model(X)
        else:
            predictions = self.ensemble_model.predict(X)

        if not return_uncertainty:
            return predictions

        # Calculate uncertainty
        individual_predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            individual_predictions.append(pred)

        individual_predictions = np.array(individual_predictions)

        if self.config.task_type == "classification":
            # Use prediction variance as uncertainty
            uncertainty = np.var(individual_predictions, axis=0)
        else:
            # Use standard deviation for regression
            uncertainty = np.std(individual_predictions, axis=0)

        return predictions, uncertainty

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classification"""
        if self.config.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if hasattr(self.ensemble_model, "predict_proba"):
            return self.ensemble_model.predict_proba(X)

        # Average probabilities from models that support it
        proba_predictions = []
        weights = []

        for model_id, model in self.models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                proba_predictions.append(proba)
                weights.append(self.model_metadata[model_id].weight)

        if not proba_predictions:
            raise ValueError("No models support probability prediction")

        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        avg_proba = np.average(proba_predictions, weights=weights, axis=0)

        return avg_proba

    def explain_prediction(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> PredictionResult:
        """Explain ensemble prediction with model contributions"""
        # Get prediction
        prediction = self.predict(X)

        # Get individual model predictions and contributions
        model_contributions = {}
        individual_predictions = {}

        for model_id, model in self.models.items():
            pred = model.predict(X)
            individual_predictions[model_id] = pred
            weight = self.model_metadata[model_id].weight
            model_contributions[model_id] = float(weight)

        # Calculate confidence
        if self.config.task_type == "classification" and hasattr(self, "predict_proba"):
            proba = self.predict_proba(X)
            confidence = float(np.max(proba))
        else:
            # Use prediction variance as inverse confidence
            preds = np.array(list(individual_predictions.values()))
            variance = np.var(preds)
            confidence = float(1.0 / (1.0 + variance))

        # Build explanation
        explanation = {
            "ensemble_type": self.config.ensemble_type,
            "n_models": len(self.models),
            "individual_predictions": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in individual_predictions.items()
            },
            "model_weights": {
                k: self.model_metadata[k].weight for k in self.models.keys()
            },
            "diversity_score": self.compute_model_diversity(individual_predictions),
        }

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            model_contributions=model_contributions,
            uncertainty=1.0 - confidence,
            explanation=explanation,
        )

    def update_model_weights(self, performance_data: Dict[str, float]) -> None:
        """Update model weights based on recent performance"""
        # Normalize performance scores
        total_score = sum(performance_data.values())

        if total_score > 0:
            for model_id, score in performance_data.items():
                if model_id in self.model_metadata:
                    # Exponential moving average update
                    alpha = 0.3
                    old_weight = self.model_metadata[model_id].weight
                    new_weight = score / total_score
                    self.model_metadata[model_id].weight = (
                        alpha * new_weight + (1 - alpha) * old_weight
                    )

        logger.info(
            "Updated model weights",
            weights={k: v.weight for k, v in self.model_metadata.items()},
        )

    def add_calibration(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Add probability calibration to ensemble"""
        if self.config.task_type != "classification":
            return

        from sklearn.calibration import CalibratedClassifierCV

        # Calibrate each model
        for model_id, model in self.models.items():
            if hasattr(model, "predict_proba"):
                calibrated = CalibratedClassifierCV(
                    model,
                    method=self.config.calibration_method or "isotonic",
                    cv="prefit",
                )
                calibrated.fit(X_cal, y_cal)
                self.calibrators[model_id] = calibrated

        logger.info("Added calibration to models", n_calibrated=len(self.calibrators))

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get aggregated feature importance from ensemble"""
        importance_scores = {}

        for model_id, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                weight = self.model_metadata[model_id].weight

                for i, imp in enumerate(importances):
                    feature = feature_names[i] if feature_names else f"feature_{i}"
                    if feature not in importance_scores:
                        importance_scores[feature] = 0
                    importance_scores[feature] += imp * weight

        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}

        self.feature_importance = importance_scores
        return importance_scores

    async def optimize_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Optimize ensemble hyperparameters using Optuna"""

        def objective(trial):
            # Suggest ensemble configuration
            ensemble_type = trial.suggest_categorical(
                "ensemble_type", ["voting", "stacking", "boosting", "dynamic"]
            )

            # Suggest number of models
            n_models = trial.suggest_int("n_models", 3, 20)

            # Suggest diversity threshold
            diversity_threshold = trial.suggest_float("diversity_threshold", 0.1, 0.5)

            # Create new config
            config = EnsembleConfig(
                ensemble_type=ensemble_type,
                task_type=self.config.task_type,
                combination_method=self.config.combination_method,
                diversity_threshold=diversity_threshold,
                max_models=n_models,
            )

            # Create temporary ensemble
            temp_ensemble = ModelEnsemble(config)

            # Copy models
            for model_id, model in list(self.models.items())[:n_models]:
                temp_ensemble.add_model(model, model_id, self.model_metadata[model_id])

            # Train and evaluate
            loop = asyncio.get_event_loop()
            performance = loop.run_until_complete(
                temp_ensemble.train_ensemble(X_train, y_train, X_val, y_val)
            )

            return performance.overall_score

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        logger.info(
            "Ensemble optimization completed",
            best_params=best_params,
            best_score=best_score,
        )

        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_history": [
                {"params": t.params, "value": t.value} for t in study.trials
            ],
        }

    def save_ensemble(self, path: Path) -> None:
        """Save ensemble to disk"""
        ensemble_data = {
            "config": self.config,
            "models": {},
            "metadata": self.model_metadata,
            "feature_importance": self.feature_importance,
            "performance_history": self.performance_history,
        }

        # Save each model
        for model_id, model in self.models.items():
            model_path = path / f"model_{model_id}.pkl"
            joblib.dump(model, model_path)
            ensemble_data["models"][model_id] = str(model_path)

        # Save ensemble data
        with open(path / "ensemble.json", "w") as f:
            json.dump(
                ensemble_data,
                f,
                default=lambda x: (
                    str(x) if isinstance(x, (datetime, Path)) else x.__dict__
                ),
                indent=2,
            )

        logger.info("Saved ensemble", path=path)

    @classmethod
    def load_ensemble(cls, path: Path) -> "ModelEnsemble":
        """Load ensemble from disk"""
        # Load ensemble data
        with open(path / "ensemble.json", "r") as f:
            ensemble_data = json.load(f)

        # Recreate config
        config = EnsembleConfig(**ensemble_data["config"])
        ensemble = cls(config)

        # Load models
        for model_id, model_path in ensemble_data["models"].items():
            model = joblib.load(model_path)
            metadata = ModelMetadata(**ensemble_data["metadata"][model_id])
            ensemble.add_model(model, model_id, metadata)

        ensemble.feature_importance = ensemble_data.get("feature_importance", {})

        logger.info("Loaded ensemble", path=path)
        return ensemble


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = EnsembleConfig(
        ensemble_type="stacking",
        task_type="classification",
        combination_method="meta-learner",
        cv_folds=5,
        optimization_metric="accuracy",
        diversity_threshold=0.3,
        dynamic_weighting=True,
    )

    # Create ensemble
    ensemble = ModelEnsemble(config)

    # Example: Add models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    # Add diverse models
    rf_model = RandomForestClassifier(n_estimators=100)
    svm_model = SVC(probability=True)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50))

    # Add with metadata
    ensemble.add_model(
        rf_model,
        "rf_001",
        ModelMetadata(
            model_id="rf_001",
            model_type="RandomForest",
            training_accuracy=0.92,
            validation_accuracy=0.88,
        ),
    )

    ensemble.add_model(
        svm_model,
        "svm_001",
        ModelMetadata(
            model_id="svm_001",
            model_type="SVM",
            training_accuracy=0.90,
            validation_accuracy=0.86,
        ),
    )

    ensemble.add_model(
        mlp_model,
        "mlp_001",
        ModelMetadata(
            model_id="mlp_001",
            model_type="NeuralNetwork",
            training_accuracy=0.91,
            validation_accuracy=0.87,
        ),
    )

    logger.info("ModelEnsemble example completed")

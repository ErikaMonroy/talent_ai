#!/usr/bin/env python3
"""
XGBoost Model for TalentAI Area Recommendation System

This module implements an XGBoost classifier for predicting student
area of knowledge based on ICFES scores and dimensional assessments.

Features:
- Advanced hyperparameter optimization with Optuna
- Early stopping and learning curve analysis
- SHAP values for model interpretability
- Feature importance with multiple methods
- Cross-validation with stratification
- GPU acceleration support (if available)
- Model persistence and deployment utilities

"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor, evaluate_model_performance
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Using GridSearchCV for hyperparameter tuning.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Feature importance will use built-in XGBoost methods.")

class XGBoostModel:
    """
    XGBoost model wrapper for the TalentAI project.
    
    This class provides a complete implementation of XGBoost
    with advanced optimization, interpretability, and evaluation capabilities.
    """
    
    def __init__(self, random_state=42, use_gpu=False):
        """
        Initialize the XGBoost model.
        
        Args:
            random_state (int): Random seed for reproducibility
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.model = self.create_model()  # Initialize with default parameters
        self.best_params = None
        self.is_trained = False
        self.feature_names = None
        self.performance_metrics = {}
        self.feature_importance_df = None
        self.training_history = None
        self.shap_explainer = None
        self.shap_values = None
        
    def create_model(self, **kwargs):
        """
        Create an XGBoost model with specified parameters.
        
        Args:
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            XGBClassifier: Configured model
        """
        default_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'verbosity': 0,
            'enable_categorical': False
        }
        
        # GPU configuration
        if self.use_gpu:
            try:
                default_params['tree_method'] = 'gpu_hist'
                default_params['gpu_id'] = 0
                print("GPU acceleration enabled for XGBoost")
            except Exception as e:
                print(f"GPU not available, using CPU: {e}")
                self.use_gpu = False
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
        return self.model
    
    def optuna_hyperparameter_tuning(self, X_train, y_train, X_val=None, y_val=None, 
                                   n_trials=10, cv=3):
        """
        Perform FAST hyperparameter tuning using Optuna.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            X_val (array): Validation features (optional)
            y_val (array): Validation targets (optional)
            n_trials (int): Number of optimization trials (reduced for speed)
            cv (int): Number of cross-validation folds (reduced for speed)
            
        Returns:
            dict: Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using default parameters.")
            return self._default_hyperparameters()
        
        print("=== FAST XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA ===")
        print(f"⚡ OPTIMIZED FOR SPEED: {n_trials} trials, {cv}-fold CV")
        
        def objective(trial):
            # Define REDUCED parameter search space for faster training
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Reduced range
                'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),  # Narrower range
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Narrower range
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Narrower range
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }
            
            # Create model with trial parameters
            model = self.create_model(**params)
            
            # Use validation set if provided, otherwise use cross-validation
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=50,
                         verbose=False)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='macro')
            else:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=cv, scoring='f1_macro')
                score = cv_scores.mean()
            
            return score
        
        # Create and run study
        study = optuna.create_study(direction='maximize', 
                                  sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best F1-macro score: {study.best_value:.4f}")
        
        return self.best_params
    
    def _default_hyperparameters(self):
        """
        Return default hyperparameters for XGBoost.
        
        Returns:
            dict: Default parameters
        """
        return {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, 
              tune_hyperparameters=True, early_stopping_rounds=50):
        """
        Train the XGBoost model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            X_val (array): Validation features (optional)
            y_val (array): Validation targets (optional)
            feature_names (list): Names of features
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            early_stopping_rounds (int): Early stopping patience
            
        Returns:
            XGBClassifier: Trained model
        """
        print("=== TRAINING XGBOOST MODEL ===")
        
        self.feature_names = feature_names
        
        if tune_hyperparameters:
            # Perform hyperparameter tuning
            self.optuna_hyperparameter_tuning(X_train, y_train, X_val, y_val)
            # Create model with best parameters
            if self.best_params:
                self.create_model(**self.best_params)
            else:
                # Fallback to default parameters if tuning failed
                default_params = self._default_hyperparameters()
                self.create_model(**default_params)
        else:
            # Use default parameters
            default_params = self._default_hyperparameters()
            self.create_model(**default_params)
        
        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'validation']
        else:
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        # Train the model
        fit_params = {
            'eval_set': eval_set,
            'verbose': False
        }
        
        # Only add early_stopping_rounds if validation set is provided
        if X_val is not None and y_val is not None and early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_trained = True
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        # Cross-validation evaluation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_macro')
        print(f"Cross-validation F1-macro scores: {cv_scores}")
        print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self.model
    
    def _calculate_feature_importance(self):
        """
        Calculate and store feature importance using multiple methods.
        """
        if not self.is_trained or self.feature_names is None:
            return
        
        # Get different types of feature importance
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {'feature': self.feature_names}
        
        for imp_type in importance_types:
            try:
                importance = self.model.get_booster().get_score(importance_type=imp_type)
                # Ensure all features are included (some might have 0 importance)
                imp_values = [importance.get(f'f{i}', 0) for i in range(len(self.feature_names))]
                importance_data[f'importance_{imp_type}'] = imp_values
            except Exception as e:
                print(f"Could not calculate {imp_type} importance: {e}")
        
        self.feature_importance_df = pd.DataFrame(importance_data)
        
        # Sort by gain importance (most informative)
        if 'importance_gain' in self.feature_importance_df.columns:
            self.feature_importance_df = self.feature_importance_df.sort_values(
                'importance_gain', ascending=False
            )
            
            print(f"\nTop 5 most important features (by gain):")
            for idx, row in self.feature_importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance_gain']:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): Test targets
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        self.performance_metrics = evaluate_model_performance(
            y_test, y_pred, "XGBoost"
        )
        
        return self.performance_metrics
    
    def plot_feature_importance(self, importance_type='gain', top_n=15, figsize=(12, 8)):
        """
        Plot feature importance.
        
        Args:
            importance_type (str): Type of importance ('gain', 'weight', 'cover')
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if self.feature_importance_df is None:
            print("Feature importance not calculated. Train the model first.")
            return
        
        importance_col = f'importance_{importance_type}'
        if importance_col not in self.feature_importance_df.columns:
            print(f"Importance type '{importance_type}' not available.")
            return
        
        plt.figure(figsize=figsize)
        
        # Sort and get top features
        top_features = self.feature_importance_df.nlargest(top_n, importance_col)
        
        # Create horizontal bar plot
        sns.barplot(data=top_features, x=importance_col, y='feature', palette='viridis')
        plt.title(f'XGBoost - Feature Importance ({importance_type.title()})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'Feature Importance ({importance_type.title()})', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels on bars
        for i, v in enumerate(top_features[importance_col]):
            plt.text(v + max(top_features[importance_col]) * 0.01, i, f'{v:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def plot_training_curves(self, figsize=(15, 5)):
        """
        Plot training and validation curves.
        
        Args:
            figsize (tuple): Figure size
        """
        if self.training_history is None:
            print("Training history not available. Train the model with validation set.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training curves for each metric
        for eval_name, metrics in self.training_history.items():
            for metric_name, values in metrics.items():
                axes[0].plot(values, label=f'{eval_name}_{metric_name}')
        
        axes[0].set_title('XGBoost - Training Curves', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Boosting Rounds', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot learning curve (if validation data available)
        if len(self.training_history) > 1:
            train_scores = list(self.training_history.values())[0]
            val_scores = list(self.training_history.values())[1]
            
            train_metric = list(train_scores.values())[0]
            val_metric = list(val_scores.values())[0]
            
            axes[1].plot(train_metric, label='Training', color='blue')
            axes[1].plot(val_metric, label='Validation', color='red')
            axes[1].set_title('XGBoost - Learning Curve', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Boosting Rounds', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Validation data\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12)
            axes[1].set_title('Validation Curve', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, figsize=(12, 10)):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            X_test (array): Test features
            y_test (array): Test targets
            figsize (tuple): Figure size
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting confusion matrix")
            
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('XGBoost - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Area', fontsize=12)
        plt.ylabel('True Area', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def calculate_shap_values(self, X_sample, sample_size=100):
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            X_sample (array): Sample data for SHAP calculation
            sample_size (int): Number of samples to use
            
        Returns:
            array: SHAP values
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install shap package for interpretability.")
            return None
            
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating SHAP values")
        
        # Sample data if too large
        if len(X_sample) > sample_size:
            indices = np.random.choice(len(X_sample), sample_size, replace=False)
            X_sample = X_sample[indices]
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        return self.shap_values
    
    def plot_shap_summary(self, X_sample, sample_size=100, figsize=(12, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            X_sample (array): Sample data for SHAP calculation
            sample_size (int): Number of samples to use
            figsize (tuple): Figure size
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install shap package for interpretability.")
            return
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X_sample, sample_size)
        
        if self.shap_values is None:
            return
        
        plt.figure(figsize=figsize)
        
        # For multiclass, use the first class or average
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[0]  # First class
        else:
            shap_values_plot = self.shap_values
        
        shap.summary_plot(shap_values_plot, X_sample[:len(shap_values_plot)], 
                         feature_names=self.feature_names, show=False)
        plt.title('XGBoost - SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self):
        """
        Get detailed information about the trained model.
        
        Returns:
            dict: Model information
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        info = {
            'model_type': 'XGBoost',
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'n_classes': len(self.model.classes_) if self.model else 'Unknown',
            'n_estimators': self.model.n_estimators if self.model else 'Unknown',
            'best_iteration': getattr(self.model, 'best_iteration', 'Not available'),
            'use_gpu': self.use_gpu,
            'performance_metrics': self.performance_metrics
        }
        
        if self.best_params:
            info['best_parameters'] = self.best_params
        
        if hasattr(self.model, 'feature_importances_'):
            info['has_feature_importance'] = True
        
        return info
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'feature_importance_df': self.feature_importance_df,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance_df = model_data['feature_importance_df']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        self.use_gpu = model_data.get('use_gpu', False)
        
        print(f"Model loaded from: {filepath}")

def main():
    """
    Main function to demonstrate XGBoost model usage.
    """
    print("=== XGBOOST MODEL DEMO ===")
    
    # Initialize data preprocessor
    dataset_path = "../dataset_estudiantes.csv"
    preprocessor = DataPreprocessor(dataset_path)
    
    # Preprocess data
    data = preprocessor.full_preprocessing_pipeline()
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        data['X_train_scaled'], data['y_train'], 
        test_size=0.2, random_state=42, stratify=data['y_train']
    )
    
    # Initialize and train model
    xgb_model = XGBoostModel(use_gpu=False)  # Set to True if GPU available
    
    # Train with hyperparameter tuning and validation
    xgb_model.train(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        feature_names=data['feature_columns'],
        tune_hyperparameters=True
    )
    
    # Evaluate model
    performance = xgb_model.evaluate(data['X_test_scaled'], data['y_test'])
    
    # Plot feature importance
    xgb_model.plot_feature_importance(importance_type='gain')
    
    # Plot training curves
    xgb_model.plot_training_curves()
    
    # Plot confusion matrix
    xgb_model.plot_confusion_matrix(data['X_test_scaled'], data['y_test'])
    
    # Calculate and plot SHAP values (if available)
    if SHAP_AVAILABLE:
        xgb_model.plot_shap_summary(data['X_test_scaled'])
    
    # Save model
    xgb_model.save_model('xgboost_model.joblib')
    
    # Print model info
    model_info = xgb_model.get_model_info()
    print("\n=== MODEL INFO ===")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    return xgb_model, performance

if __name__ == "__main__":
    model, metrics = main()
#!/usr/bin/env python3
"""
Logistic Regression Model for TalentAI Area Recommendation System

This module implements a Logistic Regression classifier for predicting student
area of knowledge based on ICFES scores and dimensional assessments.

Features:
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Comprehensive performance metrics
- Model persistence (save/load)
- Feature importance analysis

"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor, evaluate_model_performance
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionModel:
    """
    Logistic Regression model wrapper for the TalentAI project.
    
    This class provides a complete implementation of Logistic Regression
    with hyperparameter tuning, evaluation, and visualization capabilities.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the Logistic Regression model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None
        self.best_model: Optional[LogisticRegression] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.is_trained: bool = False
        self.feature_names: Optional[List[str]] = None
        self.performance_metrics: Dict[str, Any] = {}
        
    def create_model(self, **kwargs: Any) -> LogisticRegression:
        """
        Create a Logistic Regression model with specified parameters.
        
        Args:
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            LogisticRegression: Configured model
        """
        default_params = {
            'random_state': self.random_state,
            'max_iter': 1000,
            'multi_class': 'ovr',  # One-vs-Rest for multiclass
            'solver': 'liblinear'  # Good for small datasets
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        self.model = LogisticRegression(**default_params)
        return self.model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, scoring: str = 'f1_macro', n_jobs: int = -1) -> GridSearchCV:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for optimization
            n_jobs (int): Number of parallel jobs
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        print("=== LOGISTIC REGRESSION HYPERPARAMETER TUNING ===")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l1', 'l2'],  # Regularization type
            'solver': ['liblinear', 'saga'],  # Solvers that support both L1 and L2
            'max_iter': [1000, 2000]
        }
        
        # Create base model
        base_model = LogisticRegression(
            random_state=self.random_state,
            multi_class='ovr'
        )
        
        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        print(f"Starting grid search with {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['max_iter'])} combinations...")
        
        self.grid_search.fit(X_train, y_train)
        
        # Get best model
        self.best_model = self.grid_search.best_estimator_
        self.model = self.best_model
        
        print(f"Best parameters: {self.grid_search.best_params_}")
        print(f"Best cross-validation score: {self.grid_search.best_score_:.4f}")
        
        return self.grid_search
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None, tune_hyperparameters: bool = True) -> LogisticRegression:
        """
        Train the Logistic Regression model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            feature_names (list): Names of features
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            LogisticRegression: Trained model
        """
        print("=== TRAINING LOGISTIC REGRESSION MODEL ===")
        
        self.feature_names = feature_names
        
        if tune_hyperparameters:
            # Perform hyperparameter tuning
            self.hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters
            self.create_model()
            if self.model is not None:
                self.model.fit(X_train, y_train)
            else:
                raise ValueError("Model creation failed")
        
        self.is_trained = True
        
        # Cross-validation evaluation
        if self.model is not None:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_macro')
            print(f"Cross-validation F1-macro scores: {cv_scores}")
            print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            return self.model
        else:
            raise ValueError("Model training failed")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if self.model is not None:
            return self.model.predict(X_test)
        else:
            raise ValueError("Model is not initialized")
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if self.model is not None:
            return self.model.predict_proba(X_test)
        else:
            raise ValueError("Model is not initialized")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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
            y_test, y_pred, "Logistic Regression"
        )
        
        return self.performance_metrics
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)) -> Optional[pd.DataFrame]:
        """
        Plot feature importance based on model coefficients.
        
        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")
            
        if self.feature_names is None:
            print("Feature names not available for plotting")
            return
        
        # Get average absolute coefficients across all classes
        if self.model is not None and hasattr(self.model, 'coef_'):
            coef_abs = np.abs(self.model.coef_).mean(axis=0)
        else:
            raise ValueError("Model coefficients not available")
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coef_abs
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=figsize)
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Logistic Regression - Feature Importance\n(Average Absolute Coefficients)', fontsize=14, fontweight='bold')
        plt.xlabel('Average Absolute Coefficient', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, figsize: Tuple[int, int] = (12, 10)) -> np.ndarray:
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
        plt.title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Area', fontsize=12)
        plt.ylabel('True Area', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'best_model': self.best_model,
            'grid_search': self.grid_search,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_model = model_data['best_model']
        self.grid_search = model_data['grid_search']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model and its performance.
        
        Returns:
            dict: Model summary
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
            
        summary = {
            'model_type': 'Logistic Regression',
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'n_classes': len(self.model.classes_) if self.model is not None else 'Unknown',
            'performance_metrics': self.performance_metrics
        }
        
        if self.grid_search:
            summary['best_parameters'] = self.grid_search.best_params_
            summary['best_cv_score'] = self.grid_search.best_score_
            
        return summary

def main():
    """
    Main function to demonstrate Logistic Regression model usage.
    """
    print("=== LOGISTIC REGRESSION MODEL DEMO ===")
    
    # Initialize data preprocessor
    dataset_path = "../dataset_estudiantes.csv"
    preprocessor = DataPreprocessor(dataset_path)
    
    # Preprocess data
    data = preprocessor.full_preprocessing_pipeline()
    
    # Initialize and train model
    lr_model = LogisticRegressionModel()
    
    # Train with hyperparameter tuning
    lr_model.train(
        data['X_train_scaled'], 
        data['y_train'], 
        feature_names=data['feature_columns'],
        tune_hyperparameters=True
    )
    
    # Evaluate model
    performance = lr_model.evaluate(data['X_test_scaled'], data['y_test'])
    
    # Plot feature importance
    lr_model.plot_feature_importance()
    
    # Plot confusion matrix
    lr_model.plot_confusion_matrix(data['X_test_scaled'], data['y_test'])
    
    # Save model
    lr_model.save_model('logistic_regression_model.joblib')
    
    # Print summary
    summary = lr_model.get_model_summary()
    print("\n=== MODEL SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return lr_model, performance

if __name__ == "__main__":
    model, metrics = main()
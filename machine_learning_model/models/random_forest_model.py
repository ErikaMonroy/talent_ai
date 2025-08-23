#!/usr/bin/env python3
"""
Random Forest Model for TalentAI Area Recommendation System

This module implements a Random Forest classifier for predicting student
area of knowledge based on ICFES scores and dimensional assessments.

Features:
- Advanced hyperparameter tuning with RandomizedSearchCV
- Feature importance analysis with multiple methods
- Out-of-bag (OOB) score evaluation
- Tree visualization capabilities
- Cross-validation for robust evaluation
- Model persistence and interpretability

"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor, evaluate_model_performance
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
warnings.filterwarnings('ignore')

class RandomForestModel:
    """
    Random Forest model wrapper for the TalentAI project.
    
    This class provides a complete implementation of Random Forest
    with advanced hyperparameter tuning, feature importance analysis,
    and comprehensive evaluation capabilities.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the Random Forest model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.best_model: Optional[RandomForestClassifier] = None
        self.randomized_search: Optional[RandomizedSearchCV] = None
        self.is_trained: bool = False
        self.feature_names: Optional[List[str]] = None
        self.performance_metrics: Dict[str, Any] = {}
        self.feature_importance_df: Optional[pd.DataFrame] = None
        
    def create_model(self, **kwargs: Any) -> RandomForestClassifier:
        """
        Create a Random Forest model with specified parameters.
        
        Args:
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Configured model
        """
        default_params = {
            'random_state': self.random_state,
            'n_jobs': -1,  # Use all available cores
            'oob_score': True,  # Enable out-of-bag scoring
            'bootstrap': True,  # Enable bootstrap sampling
            'class_weight': 'balanced'  # Handle class imbalance
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
        return self.model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, scoring: str = 'f1_macro', 
                            n_iter: int = 100, n_jobs: int = -1) -> RandomizedSearchCV:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for optimization
            n_iter (int): Number of parameter combinations to try
            n_jobs (int): Number of parallel jobs
            
        Returns:
            RandomizedSearchCV: Fitted randomized search object
        """
        print("=== RANDOM FOREST HYPERPARAMETER TUNING ===")
        
        # Define parameter distribution for randomized search
        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 12],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'max_leaf_nodes': [None, 50, 100, 200]
        }
        
        # Create base model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True,
            class_weight='balanced'
        )
        
        # Perform randomized search
        self.randomized_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=self.random_state
        )
        
        print(f"Starting randomized search with {n_iter} iterations...")
        
        self.randomized_search.fit(X_train, y_train)
        
        # Get best model
        self.best_model = self.randomized_search.best_estimator_
        self.model = self.best_model
        
        print(f"Best parameters: {self.randomized_search.best_params_}")
        print(f"Best cross-validation score: {self.randomized_search.best_score_:.4f}")
        
        # Print OOB score if available
        if self.model is not None and hasattr(self.model, 'oob_score_'):
            print(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        return self.randomized_search
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None, tune_hyperparameters: bool = True) -> Optional[RandomForestClassifier]:
        """
        Train the Random Forest model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            feature_names (list): Names of features
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("=== TRAINING RANDOM FOREST MODEL ===")
        
        self.feature_names = feature_names
        
        if tune_hyperparameters:
            # Perform hyperparameter tuning
            self.hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters
            self.create_model()
            if self.model is not None:
                self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Cross-validation evaluation
        if self.model is not None:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_macro')
            print(f"Cross-validation F1-macro scores: {cv_scores}")
            print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self.model
    
    def _calculate_feature_importance(self) -> None:
        """
        Calculate and store feature importance.
        """
        if not self.is_trained or self.feature_names is None or self.model is None:
            return
            
        # Get feature importance from the model
        importance = self.model.feature_importances_
        
        # Create feature importance dataframe
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 most important features:")
        for idx, row in self.feature_importance_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get prediction probabilities.
        
        Args:
            X_test (array): Test features
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict_proba(X_test)
    
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
            y_test, y_pred, "Random Forest"
        )
        
        # Add OOB score if available
        if self.model is not None and hasattr(self.model, 'oob_score_'):
            self.performance_metrics['oob_score'] = self.model.oob_score_
            print(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        return self.performance_metrics
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (12, 8)) -> Optional[pd.DataFrame]:
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if self.feature_importance_df is None:
            print("Feature importance not calculated. Train the model first.")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance_df.head(top_n)
        
        # Create horizontal bar plot
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Random Forest - Feature Importance\n(Gini Importance)', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return self.feature_importance_df
    
    def plot_trees_depth_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[List[int]]:
        """
        Plot distribution of tree depths in the forest.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before plotting tree depths")
        
        # Get tree depths
        tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        
        plt.figure(figsize=figsize)
        plt.hist(tree_depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Random Forest - Distribution of Tree Depths', fontsize=14, fontweight='bold')
        plt.xlabel('Tree Depth', fontsize=12)
        plt.ylabel('Number of Trees', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_depth = float(np.mean(tree_depths))
        plt.axvline(mean_depth, color='red', linestyle='--', 
                   label=f'Mean Depth: {mean_depth:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return tree_depths
    
    def plot_single_tree(self, tree_index: int = 0, max_depth: int = 3, figsize: Tuple[int, int] = (20, 10)) -> None:
        """
        Visualize a single tree from the forest.
        
        Args:
            tree_index (int): Index of tree to visualize
            max_depth (int): Maximum depth to display
            figsize (tuple): Figure size
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before plotting trees")
            
        if tree_index >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_index} out of range. Forest has {len(self.model.estimators_)} trees.")
        
        plt.figure(figsize=figsize)
        
        tree = self.model.estimators_[tree_index]
        
        plot_tree(tree, 
                 feature_names=self.feature_names,
                 class_names=[str(i) for i in range(len(self.model.classes_))],
                 filled=True,
                 rounded=True,
                 max_depth=max_depth,
                 fontsize=10)
        
        plt.title(f'Random Forest - Tree {tree_index} (Max Depth: {max_depth})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
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
        plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Area', fontsize=12)
        plt.ylabel('True Area', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trees in the forest.
        
        Returns:
            dict: Tree statistics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting tree statistics")
        
        tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in self.model.estimators_]
        tree_leaves = [tree.tree_.n_leaves for tree in self.model.estimators_]
        
        stats = {
            'n_trees': len(self.model.estimators_),
            'depth_stats': {
                'mean': np.mean(tree_depths),
                'std': np.std(tree_depths),
                'min': np.min(tree_depths),
                'max': np.max(tree_depths)
            },
            'nodes_stats': {
                'mean': np.mean(tree_nodes),
                'std': np.std(tree_nodes),
                'min': np.min(tree_nodes),
                'max': np.max(tree_nodes)
            },
            'leaves_stats': {
                'mean': np.mean(tree_leaves),
                'std': np.std(tree_leaves),
                'min': np.min(tree_leaves),
                'max': np.max(tree_leaves)
            }
        }
        
        return stats
    
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
            'randomized_search': self.randomized_search,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'feature_importance_df': self.feature_importance_df,
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
        self.randomized_search = model_data['randomized_search']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance_df = model_data['feature_importance_df']
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
            'model_type': 'Random Forest',
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'n_classes': len(self.model.classes_) if self.model is not None else 'Unknown',
            'n_estimators': self.model.n_estimators if self.model is not None else 'Unknown',
            'performance_metrics': self.performance_metrics,
            'tree_statistics': self.get_tree_statistics() if self.is_trained else None
        }
        
        if self.randomized_search:
            summary['best_parameters'] = self.randomized_search.best_params_
            summary['best_cv_score'] = self.randomized_search.best_score_
            
        return summary

def main():
    """
    Main function to demonstrate Random Forest model usage.
    """
    print("=== RANDOM FOREST MODEL DEMO ===")
    
    # Initialize data preprocessor
    dataset_path = "../dataset_estudiantes.csv"
    preprocessor = DataPreprocessor(dataset_path)
    
    # Preprocess data
    data = preprocessor.full_preprocessing_pipeline()
    
    # Initialize and train model
    rf_model = RandomForestModel()
    
    # Train with hyperparameter tuning
    rf_model.train(
        data['X_train_scaled'], 
        data['y_train'], 
        feature_names=data['feature_columns'],
        tune_hyperparameters=True
    )
    
    # Evaluate model
    performance = rf_model.evaluate(data['X_test_scaled'], data['y_test'])
    
    # Plot feature importance
    rf_model.plot_feature_importance()
    
    # Plot tree depth distribution
    rf_model.plot_trees_depth_distribution()
    
    # Plot confusion matrix
    rf_model.plot_confusion_matrix(data['X_test_scaled'], data['y_test'])
    
    # Save model
    rf_model.save_model('random_forest_model.joblib')
    
    # Print summary
    summary = rf_model.get_model_summary()
    print("\n=== MODEL SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return rf_model, performance

if __name__ == "__main__":
    model, metrics = main()
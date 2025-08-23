#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Model for TalentAI Area Recommendation System

This module implements a K-Nearest Neighbors classifier for predicting
student area of knowledge based on ICFES scores and dimensional assessments.

Features:
- Advanced distance metrics and weighting schemes
- Optimal k selection with cross-validation
- Neighborhood analysis and visualization
- Curse of dimensionality mitigation techniques
- Distance-based feature importance
- Outlier detection and handling
- Efficient nearest neighbor search algorithms
- Model interpretability through neighbor analysis

"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from .data_preprocessing import DataPreprocessor, evaluate_model_performance
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from sklearn.neighbors import LocalOutlierFactor
    LOF_AVAILABLE = True
except ImportError:
    LOF_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Using t-SNE for dimensionality reduction visualization.")

class KNNModel:
    """
    K-Nearest Neighbors model wrapper for the TalentAI project.
    
    This class provides a comprehensive implementation of KNN
    with advanced distance metrics, optimal k selection, and
    neighborhood analysis capabilities.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the KNN model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[KNeighborsClassifier] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.performance_metrics: Dict[str, Any] = {}
        self.optimal_k: Optional[int] = None
        self.distance_analysis: Optional[Dict[str, Any]] = None
        self.neighbor_analysis: Optional[List[Dict[str, Any]]] = None
        self.outlier_detector: Optional[Any] = None
        
        # Set random seed
        np.random.seed(random_state)
    
    def create_model(self, **kwargs: Any) -> KNeighborsClassifier:
        """
        Create a KNN model with specified parameters.
        
        Args:
            **kwargs: Additional parameters for KNeighborsClassifier
            
        Returns:
            KNeighborsClassifier: Configured model
        """
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'metric': 'minkowski',
            'p': 2,  # Euclidean distance
            'n_jobs': -1
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        self.model = KNeighborsClassifier(**default_params)
        return self.model
    
    def find_optimal_k(self, X_train: np.ndarray, y_train: np.ndarray, k_range: Optional[List[int]] = None, cv: int = 5, scoring: str = 'f1_macro', plot_results: bool = False) -> int:
        """
        Find optimal k value using cross-validation.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            k_range (list): Range of k values to test
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            plot_results (bool): Whether to generate visualization plots (default: False for production)
            
        Returns:
            int: Optimal k value
        """
        print("=== FINDING OPTIMAL K VALUE ===")
        
        if k_range is None:
            # Default k range based on dataset size
            max_k = min(50, len(X_train) // 10)
            k_range = list(range(1, max_k + 1, 2))  # Odd numbers only
        
        # Test different k values
        k_scores = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring=scoring)
            k_scores.append(scores.mean())
            print(f"k={k}: {scoring}={scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Find optimal k
        optimal_idx = np.argmax(k_scores)
        self.optimal_k = k_range[optimal_idx]
        
        print(f"\nOptimal k: {self.optimal_k} with {scoring}: {k_scores[optimal_idx]:.4f}")
        
        # Plot k vs performance (only if requested - disabled in production)
        if plot_results:
            self._plot_k_selection(k_range, k_scores, scoring)
        
        return self.optimal_k
    
    def _plot_k_selection(self, k_range: List[int], k_scores: List[float], scoring: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot k selection curve.
        
        Args:
            k_range (list): Range of k values
            k_scores (list): Corresponding scores
            scoring (str): Scoring metric name
            figsize (tuple): Figure size
        """
        if self.optimal_k is None:
            return
            
        plt.figure(figsize=figsize)
        plt.plot(k_range, k_scores, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {self.optimal_k}')
        plt.title('KNN - Optimal k Selection', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel(f'{scoring.upper()} Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight optimal point
        optimal_idx = k_range.index(self.optimal_k)
        plt.scatter(self.optimal_k, k_scores[optimal_idx], 
                   color='red', s=100, zorder=5)
        
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, scoring: str = 'f1_macro') -> GridSearchCV:
        """
        Perform comprehensive hyperparameter tuning.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        print("=== KNN HYPERPARAMETER TUNING ===")
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': list(range(1, 31, 2)),  # Odd numbers 1-29
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            'p': [1, 2, 3]  # For minkowski metric
        }
        
        # Create base model
        base_model = KNeighborsClassifier(n_jobs=-1)
        
        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        self.grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = self.grid_search.best_estimator_
        if self.model is not None:
            self.optimal_k = self.model.n_neighbors
        
        print(f"Best parameters: {self.grid_search.best_params_}")
        print(f"Best cross-validation score: {self.grid_search.best_score_:.4f}")
        
        return self.grid_search
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None, tune_hyperparameters: bool = True, 
              find_optimal_k_only: bool = False, plot_results: bool = False) -> KNeighborsClassifier:
        """
        Train the KNN model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            feature_names (list): Names of features
            tune_hyperparameters (bool): Whether to perform full hyperparameter tuning
            find_optimal_k_only (bool): Whether to only find optimal k
            plot_results (bool): Whether to generate visualization plots (default: False for production)
            
        Returns:
            KNeighborsClassifier: Trained model
        """
        print("=== TRAINING KNN MODEL ===")
        
        self.feature_names = feature_names
        
        if find_optimal_k_only:
            # Only find optimal k, use default other parameters
            self.find_optimal_k(X_train, y_train, plot_results=plot_results)
            self.create_model(n_neighbors=self.optimal_k)
            if self.model is not None:
                self.model.fit(X_train, y_train)
        elif tune_hyperparameters:
            # Perform full hyperparameter tuning
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
        
        # Analyze training data
        self._analyze_training_data(X_train, y_train)
        
        if self.model is None:
            raise ValueError("Model training failed")
        return self.model
    
    def _analyze_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Analyze training data characteristics.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
        """
        print("\n=== TRAINING DATA ANALYSIS ===")
        
        # Dataset characteristics
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        print(f"Dataset size: {n_samples} samples, {n_features} features")
        print(f"Number of classes: {n_classes}")
        print(f"Samples per feature ratio: {n_samples/n_features:.2f}")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        
        # Distance analysis
        self._analyze_distances(X_train, y_train)
    
    def _analyze_distances(self, X_train: np.ndarray, y_train: np.ndarray, sample_size: int = 1000) -> None:
        """
        Analyze distance characteristics of the dataset.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            sample_size (int): Sample size for distance analysis
        """
        # Sample data if too large
        if len(X_train) > sample_size:
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        # Calculate pairwise distances
        distances = pairwise_distances(X_sample, metric='euclidean')
        
        # Distance statistics
        self.distance_analysis = {
            'mean_distance': np.mean(distances[distances > 0]),
            'std_distance': np.std(distances[distances > 0]),
            'min_distance': np.min(distances[distances > 0]),
            'max_distance': np.max(distances),
            'median_distance': np.median(distances[distances > 0])
        }
        
        print(f"\nDistance Analysis:")
        for key, value in self.distance_analysis.items():
            print(f"  {key}: {value:.4f}")
    
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
            proba = self.model.predict_proba(X_test)
            return np.array(proba)
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
            y_test, y_pred, "K-Nearest Neighbors"
        )
        
        return self.performance_metrics
    
    def analyze_neighbors(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze nearest neighbors for sample predictions.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            X_test (array): Test features
            n_samples (int): Number of test samples to analyze
            
        Returns:
            dict: Neighbor analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before neighbor analysis")
        
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        print("=== NEIGHBOR ANALYSIS ===")
        
        # Select random test samples
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        
        analysis_results = []
        
        for i, idx in enumerate(sample_indices):
            sample = X_test[idx:idx+1]
            
            # Find neighbors
            distances, indices = self.model.kneighbors(sample)
            
            # Get neighbor information
            neighbor_labels = y_train[indices[0]]
            neighbor_distances = distances[0]
            
            # Prediction and confidence
            prediction = self.model.predict(sample)[0]
            probabilities = self.model.predict_proba(sample)[0]
            confidence = np.max(probabilities)
            
            result = {
                'sample_index': idx,
                'prediction': prediction,
                'confidence': confidence,
                'neighbor_labels': neighbor_labels,
                'neighbor_distances': neighbor_distances,
                'neighbor_indices': indices[0]
            }
            
            analysis_results.append(result)
            
            print(f"\nSample {i+1} (Index {idx}):")
            print(f"  Prediction: {prediction} (Confidence: {confidence:.3f})")
            print(f"  Neighbor labels: {neighbor_labels}")
            print(f"  Neighbor distances: {neighbor_distances}")
        
        self.neighbor_analysis = analysis_results
        return analysis_results
    
    def detect_outliers(self, X_train: np.ndarray, contamination: Union[float, str] = 0.1) -> Optional[np.ndarray]:
        """
        Detect outliers in training data using Local Outlier Factor.
        
        Args:
            X_train (array): Training features
            contamination (float or str): Expected proportion of outliers
            
        Returns:
            array: Outlier labels (-1 for outliers, 1 for inliers)
        """
        if not LOF_AVAILABLE:
            print("Local Outlier Factor not available.")
            return None
        
        print("=== OUTLIER DETECTION ===")
        
        # Use same k as the main model if available
        n_neighbors = self.optimal_k if self.optimal_k else 20
        
        # Handle contamination parameter type
        if isinstance(contamination, str):
            contamination = float(contamination) if contamination != 'auto' else contamination
        
        self.outlier_detector = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=str(contamination),
            n_jobs=-1
        )
        
        outlier_labels = self.outlier_detector.fit_predict(X_train)
        
        n_outliers = int(np.sum(outlier_labels == -1))
        n_inliers = int(np.sum(outlier_labels == 1))
        
        print(f"Detected {n_outliers} outliers and {n_inliers} inliers")
        print(f"Outlier ratio: {n_outliers/len(X_train):.3f}")
        
        return outlier_labels
    
    def plot_distance_distribution(self, X_train: np.ndarray, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot distance distribution analysis.
        
        Args:
            X_train (array): Training features
            figsize (tuple): Figure size
        """
        if self.distance_analysis is None:
            print("Distance analysis not performed. Train the model first.")
            return
        
        # Sample data for visualization
        sample_size = min(500, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        
        # Calculate distances
        distances = pairwise_distances(X_sample, metric='euclidean')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Distance histogram
        axes[0, 0].hist(distances[distances > 0], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.distance_analysis['mean_distance'], color='red', 
                          linestyle='--', label='Mean')
        axes[0, 0].axvline(self.distance_analysis['median_distance'], color='green', 
                          linestyle='--', label='Median')
        axes[0, 0].set_title('Distance Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Euclidean Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distance boxplot
        axes[0, 1].boxplot(distances[distances > 0])
        axes[0, 1].set_title('Distance Boxplot', fontweight='bold')
        axes[0, 1].set_ylabel('Euclidean Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Nearest neighbor distances
        nn_distances = []
        for i in range(len(X_sample)):
            row_distances = distances[i]
            row_distances = row_distances[row_distances > 0]  # Exclude self
            if len(row_distances) > 0:
                nn_distances.append(np.min(row_distances))
        
        axes[1, 0].hist(nn_distances, bins=30, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Nearest Neighbor Distances', fontweight='bold')
        axes[1, 0].set_xlabel('Distance to Nearest Neighbor')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # K-distance plot (for different k values)
        k_values = [1, 3, 5, 10, 15]
        for k in k_values:
            if k < len(X_sample):
                k_distances = []
                for i in range(len(X_sample)):
                    row_distances = np.sort(distances[i])
                    if len(row_distances) > k:
                        k_distances.append(row_distances[k])  # k-th nearest neighbor
                
                axes[1, 1].plot(sorted(k_distances), label=f'k={k}')
        
        axes[1, 1].set_title('K-Distance Plot', fontweight='bold')
        axes[1, 1].set_xlabel('Points (sorted by k-distance)')
        axes[1, 1].set_ylabel('K-Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimensionality_reduction(self, X_train: np.ndarray, y_train: np.ndarray, method: str = 'pca', figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot dimensionality reduction visualization.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            method (str): Reduction method ('pca', 'tsne', 'umap')
            figsize (tuple): Figure size
        """
        # Sample data if too large
        sample_size = min(1000, len(X_train))
        if len(X_train) > sample_size:
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_sample)
        
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, 
                                 cmap='tab20', alpha=0.7, s=30)
        axes[0].set_title('PCA Visualization', fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0])
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                                 cmap='tab20', alpha=0.7, s=30)
        axes[1].set_title('t-SNE Visualization', fontweight='bold')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[1])
        
        # UMAP (if available)
        if UMAP_AVAILABLE and method == 'umap':
            umap_reducer = umap.UMAP(n_components=2, random_state=self.random_state)
            X_umap = umap_reducer.fit_transform(X_sample)
            
            scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample, 
                                     cmap='tab20', alpha=0.7, s=30)
            axes[2].set_title('UMAP Visualization', fontweight='bold')
            axes[2].set_xlabel('UMAP 1')
            axes[2].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[2])
        else:
            # Alternative: Show feature correlation heatmap
            if self.feature_names and len(self.feature_names) <= 20:  # Only for reasonable number of features
                corr_matrix = np.corrcoef(X_sample.T)
                im = axes[2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                axes[2].set_title('Feature Correlation', fontweight='bold')
                axes[2].set_xticks(range(len(self.feature_names)))
                axes[2].set_yticks(range(len(self.feature_names)))
                axes[2].set_xticklabels(self.feature_names, rotation=45)
                axes[2].set_yticklabels(self.feature_names)
                plt.colorbar(im, ax=axes[2])
            else:
                axes[2].text(0.5, 0.5, 'Too many features\nfor correlation plot', 
                           ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Feature Space Analysis', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, figsize: Tuple[int, int] = (12, 10)) -> None:
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
        plt.title('K-Nearest Neighbors - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Area', fontsize=12)
        plt.ylabel('True Area', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model and its performance.
        
        Returns:
            dict: Model summary
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
            
        summary = {
            'model_type': 'K-Nearest Neighbors',
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'optimal_k': self.optimal_k,
            'algorithm': self.model.algorithm if self.model else None,
            'metric': self.model.metric if self.model else None,
            'weights': self.model.weights if self.model else None,
            'performance_metrics': self.performance_metrics,
            'distance_analysis': self.distance_analysis
        }
        
        if self.grid_search:
            summary['best_parameters'] = self.grid_search.best_params_
            summary['best_cv_score'] = self.grid_search.best_score_
            
        return summary
    
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
            'grid_search': self.grid_search,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'optimal_k': self.optimal_k,
            'distance_analysis': self.distance_analysis,
            'neighbor_analysis': self.neighbor_analysis,
            'outlier_detector': self.outlier_detector,
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
        self.grid_search = model_data['grid_search']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.optimal_k = model_data['optimal_k']
        self.distance_analysis = model_data['distance_analysis']
        self.neighbor_analysis = model_data['neighbor_analysis']
        self.outlier_detector = model_data['outlier_detector']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")

def main():
    """
    Main function to demonstrate KNN model usage.
    """
    print("=== KNN MODEL DEMO ===")
    
    # Initialize data preprocessor
    dataset_path = "../data/datasets/dataset_estudiantes.csv"
    preprocessor = DataPreprocessor(dataset_path)
    
    # Preprocess data
    data = preprocessor.preprocess_data()
    
    # Initialize and train model
    knn_model = KNNModel()
    
    # Train with optimal k finding (faster than full hyperparameter tuning)
    knn_model.train(
        data['X_train_scaled'], 
        data['y_train'], 
        feature_names=data['feature_columns'],
        tune_hyperparameters=False,  # Set to True for full tuning
        find_optimal_k_only=True
    )
    
    # Evaluate model
    performance = knn_model.evaluate(data['X_test_scaled'], data['y_test'])
    
    # Analyze neighbors for sample predictions
    knn_model.analyze_neighbors(data['X_train_scaled'], data['y_train'], 
                               data['X_test_scaled'], n_samples=3)
    
    # Detect outliers
    outliers = knn_model.detect_outliers(data['X_train_scaled'])
    
    # Plot functions commented out for production - uncomment for development/analysis
    # knn_model.plot_distance_distribution(data['X_train_scaled'])
    # knn_model.plot_dimensionality_reduction(data['X_train_scaled'], data['y_train'])
    # knn_model.plot_confusion_matrix(data['X_test_scaled'], data['y_test'])
    
    # Save model
    knn_model.save_model('knn_model.joblib')
    
    # Print summary
    summary = knn_model.get_model_summary()
    print("\n=== MODEL SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return knn_model, performance

if __name__ == "__main__":
    model, metrics = main()
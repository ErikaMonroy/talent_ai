#!/usr/bin/env python3
"""
Visualization Utilities for TalentAI Model Comparison

This module provides advanced visualization functions for comparing
machine learning models and analyzing their performance in detail.

Features:
- Model performance comparison charts
- Learning curves and validation curves
- Feature importance analysis
- Prediction distribution analysis
- ROC curves and precision-recall curves
- Confusion matrix heatmaps
- Model interpretability visualizations
- Statistical significance testing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import label_binarize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib for all visualizations.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. SHAP visualizations will be skipped.")

class ModelVisualizationSuite:
    """
    Comprehensive visualization suite for machine learning model analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8', color_palette: str = 'husl') -> None:
        """
        Initialize the visualization suite.
        
        Args:
            figsize (tuple): Default figure size
            style (str): Matplotlib style
            color_palette (str): Seaborn color palette
        """
        self.figsize = figsize
        self.style = style
        self.color_palette = color_palette
        
        # Set style
        plt.style.use(style)
        sns.set_palette(color_palette)
        
        # Color schemes
        self.colors: List[str] = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    def plot_model_comparison_dashboard(self, results_dict: Dict[str, Dict[str, float]], training_times: Optional[Dict[str, float]] = None, 
                                      save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Create a comprehensive dashboard comparing all models.
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics as values
            training_times (dict): Dictionary with training times for each model
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        # Prepare data
        comparison_data = []
        for model_name, metrics in results_dict.items():
            data_point = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'F1-Score (Weighted)': metrics['f1_weighted']
            }
            if training_times:
                data_point['Training Time (s)'] = training_times[model_name]
            comparison_data.append(data_point)
        
        df = pd.DataFrame(comparison_data)
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(df['Model'], df['Accuracy'], color=self.colors[:len(df)])
        ax1.set_title('📊 Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, df['Accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. F1-Score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, df['F1-Score (Macro)'], width, 
                        label='Macro', color=self.colors[0], alpha=0.8)
        bars2 = ax2.bar(x + width/2, df['F1-Score (Weighted)'], width,
                        label='Weighted', color=self.colors[1], alpha=0.8)
        
        ax2.set_title('🎯 F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Model'], rotation=45)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Training time (if available)
        if training_times:
            ax3 = fig.add_subplot(gs[0, 2])
            bars = ax3.bar(df['Model'], df['Training Time (s)'], color=self.colors[2])
            ax3.set_title('⏱️ Training Time', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, df['Training Time (s)']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Training Time (s)'])*0.01,
                        f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Radar chart
        ax4 = fig.add_subplot(gs[1, :], projection='polar')
        
        metrics_for_radar = ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)']
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row['Accuracy'], row['F1-Score (Macro)'], row['F1-Score (Weighted)']]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=row['Model'], 
                    color=self.colors[i], markersize=6)
            ax4.fill(angles, values, alpha=0.25, color=self.colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics_for_radar)
        ax4.set_ylim(0, 1)
        ax4.set_title('🎯 Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 5. Performance ranking
        ax5 = fig.add_subplot(gs[2, 0])
        df_sorted = df.sort_values('F1-Score (Macro)', ascending=True)
        bars = ax5.barh(df_sorted['Model'], df_sorted['F1-Score (Macro)'], 
                       color=self.colors[:len(df_sorted)])
        ax5.set_title('🏆 Model Ranking (F1-Macro)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('F1-Score (Macro)')
        
        # Add value labels
        for bar, value in zip(bars, df_sorted['F1-Score (Macro)']):
            ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        # 6. Performance distribution
        ax6 = fig.add_subplot(gs[2, 1])
        metrics_data = [df['Accuracy'], df['F1-Score (Macro)'], df['F1-Score (Weighted)']]
        box_plot = ax6.boxplot(metrics_data, labels=['Accuracy', 'F1-Macro', 'F1-Weighted'],
                              patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], self.colors[:3]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_title('📈 Metrics Distribution', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Efficiency vs Performance
        if training_times:
            ax7 = fig.add_subplot(gs[2, 2])
            scatter = ax7.scatter(df['Training Time (s)'], df['F1-Score (Macro)'], 
                                 c=range(len(df)), cmap='viridis', s=100, alpha=0.7)
            
            # Add model labels
            for i, row in df.iterrows():
                ax7.annotate(row['Model'], (row['Training Time (s)'], row['F1-Score (Macro)']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax7.set_title('⚡ Efficiency vs Performance', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Training Time (seconds)')
            ax7.set_ylabel('F1-Score (Macro)')
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle('🎯 TalentAI - Model Comparison Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_learning_curves(self, models_dict: Dict[str, Any], X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'f1_macro',
                           train_sizes: Optional[np.ndarray] = None, save_path: Optional[str] = None, show_plot: bool = True) -> Any:
        """
        Plot learning curves for multiple models.
        
        Args:
            models_dict (dict): Dictionary of model names and model objects
            X (array): Features
            y (array): Target
            cv (int): Cross-validation folds
            scoring (str): Scoring metric
            train_sizes (array): Training set sizes to evaluate
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        n_models = len(models_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Calculate learning curve
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model.model if hasattr(model, 'model') else model,
                X, y, cv=cv, scoring=scoring, train_sizes=train_sizes,
                n_jobs=-1, random_state=42
            )
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot
            ax.plot(train_sizes_abs, train_mean, 'o-', color=self.colors[0],
                   label='Training Score', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, train_mean - train_std,
                           train_mean + train_std, alpha=0.1, color=self.colors[0])
            
            ax.plot(train_sizes_abs, val_mean, 'o-', color=self.colors[1],
                   label='Validation Score', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, val_mean - val_std,
                           val_mean + val_std, alpha=0.1, color=self.colors[1])
            
            ax.set_title(f'📈 {model_name}', fontweight='bold')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel(f'{scoring.upper()} Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('📈 Learning Curves Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_confusion_matrices(self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, class_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None, show_plot: bool = True) -> Any:
        """
        Plot confusion matrices for multiple models.
        
        Args:
            models_dict (dict): Dictionary of model names and model objects
            X_test (array): Test features
            y_test (array): Test targets
            class_names (list): Names of classes
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        n_models = len(models_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f'🔍 {model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('🔍 Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_feature_importance_comparison(self, models_dict: Dict[str, Any], feature_names: List[str],
                                         top_n: int = 10, save_path: Optional[str] = None, show_plot: bool = True) -> Tuple[Any, pd.DataFrame]:
        """
        Compare feature importance across models that support it.
        
        Args:
            models_dict (dict): Dictionary of model names and model objects
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        # Collect feature importances
        importance_data = {}
        
        for model_name, model in models_dict.items():
            if hasattr(model, 'model'):
                if hasattr(model.model, 'feature_importances_'):
                    importance_data[model_name] = model.model.feature_importances_
                elif hasattr(model.model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importance_data[model_name] = np.abs(model.model.coef_).mean(axis=0)
        
        if not importance_data:
            print("No models with feature importance found.")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        
        # Get top features (by average importance)
        importance_df['Average'] = importance_df.mean(axis=1)
        top_features = importance_df.nlargest(top_n, 'Average').drop('Average', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create grouped bar plot
        x = np.arange(len(top_features))
        width = 0.8 / len(top_features.columns)
        
        for i, model_name in enumerate(top_features.columns):
            ax.bar(x + i * width, top_features[model_name], width,
                  label=model_name, color=self.colors[i], alpha=0.8)
        
        ax.set_title(f'🎨 Top {top_n} Feature Importance Comparison', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_xticks(x + width * (len(top_features.columns) - 1) / 2)
        ax.set_xticklabels(top_features.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance comparison saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig, top_features
    
    def plot_prediction_distribution(self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, 
                                   save_path: Optional[str] = None, show_plot: bool = True) -> Any:
        """
        Plot prediction distribution and confidence for each model.
        
        Args:
            models_dict (dict): Dictionary of model names and model objects
            X_test (array): Test features
            y_test (array): Test targets
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        n_models = len(models_dict)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get predictions and probabilities
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                confidence = np.max(y_proba, axis=1)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                y_proba = model.model.predict_proba(X_test)
                confidence = np.max(y_proba, axis=1)
            else:
                confidence = np.ones(len(y_pred))  # Default confidence
            
            # Create scatter plot of predictions vs actual
            scatter = ax.scatter(y_test, y_pred, c=confidence, cmap='viridis', 
                               alpha=0.6, s=30)
            
            # Add diagonal line (perfect predictions)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_title(f'🎯 {model_name} Predictions', fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            
            # Add colorbar for confidence
            plt.colorbar(scatter, ax=ax, label='Confidence')
            
            # Calculate and display accuracy
            accuracy = np.mean(y_test == y_pred)
            ax.text(0.05, 0.95, f'Accuracy: {accuracy:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('🎯 Prediction Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_statistical_significance(self, cv_results_dict: Dict[str, Dict[str, List[float]]], alpha: float = 0.05,
                                    save_path: Optional[str] = None, show_plot: bool = True) -> Tuple[Any, np.ndarray]:
        """
        Perform statistical significance testing between models.
        
        Args:
            cv_results_dict (dict): Dictionary with CV scores for each model
            alpha (float): Significance level
            save_path (str): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        model_names = list(cv_results_dict.keys())
        n_models = len(model_names)
        
        # Create pairwise comparison matrix
        p_values = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Perform paired t-test
                scores1 = cv_results_dict[model_names[i]]['scores']
                scores2 = cv_results_dict[model_names[j]]['scores']
                
                _, p_value = stats.ttest_rel(scores1, scores2)
                p_values[i, j] = p_value
                p_values[j, i] = p_value
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask diagonal
        mask = np.eye(n_models, dtype=bool)
        
        # Create custom colormap for p-values
        colors_map = ['red' if p < alpha else 'lightblue' for p in p_values.flatten()]
        
        im = ax.imshow(p_values, cmap='RdYlBu_r', aspect='auto')
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    text = f'{p_values[i, j]:.3f}'
                    if p_values[i, j] < alpha:
                        text += '*'
                    ax.text(j, i, text, ha='center', va='center',
                           color='white' if p_values[i, j] < alpha else 'black',
                           fontweight='bold')
                else:
                    ax.text(j, i, '-', ha='center', va='center', fontweight='bold')
        
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_yticklabels(model_names)
        
        ax.set_title(f'📊 Statistical Significance Testing (α={alpha})\n* indicates significant difference',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistical significance plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig, p_values
    
    def create_interactive_dashboard(self, results_dict: Dict[str, Dict[str, float]], training_times: Optional[Dict[str, float]] = None) -> Optional[Any]:
        """
        Create an interactive dashboard using Plotly (if available).
        
        Args:
            results_dict (dict): Dictionary with model results
            training_times (dict): Dictionary with training times
        
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use matplotlib dashboard instead.")
            return None
        
        # Prepare data
        models = list(results_dict.keys())
        accuracy = [results_dict[m]['accuracy'] for m in models]
        f1_macro = [results_dict[m]['f1_macro'] for m in models]
        f1_weighted = [results_dict[m]['f1_weighted'] for m in models]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                          'Performance Radar', 'Training Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]]
        )
        
        # Accuracy bar chart
        fig.add_trace(
            go.Bar(x=models, y=accuracy, name='Accuracy', 
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=models, y=f1_macro, name='F1-Macro', 
                  marker_color='lightcoral'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=f1_weighted, name='F1-Weighted', 
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Radar chart
        for i, model in enumerate(models):
            fig.add_trace(
                go.Scatterpolar(
                    r=[accuracy[i], f1_macro[i], f1_weighted[i]],
                    theta=['Accuracy', 'F1-Macro', 'F1-Weighted'],
                    fill='toself',
                    name=model
                ),
                row=2, col=1
            )
        
        # Training time (if available)
        if training_times:
            times = [training_times[m] for m in models]
            fig.add_trace(
                go.Bar(x=models, y=times, name='Training Time', 
                      marker_color='gold'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="🎯 TalentAI - Interactive Model Comparison Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        return fig

def create_comprehensive_report(models_dict: Dict[str, Any], results_dict: Dict[str, Dict[str, float]], cv_results_dict: Dict[str, Dict[str, List[float]]], 
                              training_times: Dict[str, float], feature_names: List[str], save_path: str = "model_analysis_report.html") -> str:
    """
    Create a comprehensive HTML report with all visualizations.
    
    Args:
        models_dict (dict): Dictionary of trained models
        results_dict (dict): Dictionary of model results
        cv_results_dict (dict): Dictionary of CV results
        training_times (dict): Dictionary of training times
        feature_names (list): List of feature names
        save_path (str): Path to save the HTML report
    """
    viz_suite = ModelVisualizationSuite()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TalentAI - Model Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E86AB; text-align: center; }}
            h2 {{ color: #A23B72; border-bottom: 2px solid #A23B72; }}
            .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .best-model {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>🎯 TalentAI - Model Analysis Report</h1>
        <p><strong>Generated on:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Executive Summary</h2>
        <div class="best-model">
            <h3>🏆 Best Performing Model</h3>
    """
    
    # Find best model
    best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['f1_macro'])
    best_metrics = results_dict[best_model]
    
    html_content += f"""
            <p><strong>Model:</strong> {best_model}</p>
            <p><strong>F1-Score (Macro):</strong> {best_metrics['f1_macro']:.4f}</p>
            <p><strong>Accuracy:</strong> {best_metrics['accuracy']:.4f}</p>
            <p><strong>Training Time:</strong> {training_times[best_model]:.2f} seconds</p>
        </div>
        
        <h2>📈 Model Comparison Table</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1-Score (Macro)</th>
                <th>F1-Score (Weighted)</th>
                <th>Training Time (s)</th>
            </tr>
    """
    
    # Add model rows
    for model_name in results_dict.keys():
        metrics = results_dict[model_name]
        time_val = training_times[model_name]
        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics['accuracy']:.4f}</td>
                <td>{metrics['f1_macro']:.4f}</td>
                <td>{metrics['f1_weighted']:.4f}</td>
                <td>{time_val:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>🔍 Detailed Analysis</h2>
        <p>This report provides a comprehensive analysis of machine learning models 
        for the TalentAI knowledge area recommendation system. Each model was evaluated 
        using cross-validation and tested on a held-out test set.</p>
        
        <h3>📋 Methodology</h3>
        <ul>
            <li><strong>Data Split:</strong> 70% training, 30% testing</li>
            <li><strong>Cross-Validation:</strong> 5-fold stratified</li>
            <li><strong>Preprocessing:</strong> StandardScaler for features, LabelEncoder for target</li>
            <li><strong>Hyperparameter Tuning:</strong> GridSearchCV/RandomizedSearchCV</li>
        </ul>
        
        <h3>💡 Recommendations</h3>
        <ul>
            <li>Deploy the best performing model for production use</li>
            <li>Monitor model performance regularly</li>
            <li>Retrain models every 3-6 months</li>
            <li>Implement A/B testing for model improvements</li>
            <li>Consider ensemble methods for better performance</li>
        </ul>
        
    </body>
    </html>
    """
    
    # Save HTML report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Comprehensive report saved to: {save_path}")
    return save_path

# Example usage function
def demo_visualization_suite():
    """
    Demonstrate the visualization suite with sample data.
    """
    print("🎨 TalentAI Visualization Suite Demo")
    print("=" * 40)
    
    # Sample results data
    sample_results = {
        'Logistic Regression': {
            'accuracy': 0.85,
            'f1_macro': 0.82,
            'f1_weighted': 0.84
        },
        'Random Forest': {
            'accuracy': 0.88,
            'f1_macro': 0.86,
            'f1_weighted': 0.87
        },
        'XGBoost': {
            'accuracy': 0.90,
            'f1_macro': 0.88,
            'f1_weighted': 0.89
        }
    }
    
    sample_times = {
        'Logistic Regression': 2.5,
        'Random Forest': 15.3,
        'XGBoost': 45.7
    }
    
    # Create visualization suite
    viz_suite = ModelVisualizationSuite()
    
    # Create dashboard
    fig = viz_suite.plot_model_comparison_dashboard(
        sample_results, sample_times, show_plot=True
    )
    
    print("✅ Demo completed successfully!")
    return viz_suite

if __name__ == "__main__":
    demo_visualization_suite()
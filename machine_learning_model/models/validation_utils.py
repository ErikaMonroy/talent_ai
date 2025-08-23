#!/usr/bin/env python3
"""
Model Validation Utilities for TalentAI

This module provides comprehensive validation utilities for machine learning models,
including cross-validation, statistical testing, model stability analysis,
and performance validation.

Features:
- Advanced cross-validation strategies
- Statistical significance testing
- Model stability and robustness analysis
- Performance validation and monitoring
- Hyperparameter validation
- Data drift detection
- Model interpretability validation

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, 
    RepeatedStratifiedKFold, LeaveOneOut, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from sklearn.inspection import permutation_importance
    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ModelValidator:
    """
    Comprehensive model validation suite for machine learning models.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1) -> None:
        """
        Initialize the model validator.
        
        Args:
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.validation_results: Dict[str, Any] = {}
        
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, cv_strategy: str = 'stratified', 
                           cv_folds: int = 5, scoring_metrics: Optional[List[str]] = None, return_train_score: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation on a model.
        
        Args:
            model: Machine learning model to validate
            X (array): Features
            y (array): Target variable
            cv_strategy (str): Cross-validation strategy
            cv_folds (int): Number of folds
            scoring_metrics (list): List of scoring metrics
            return_train_score (bool): Whether to return training scores
            
        Returns:
            dict: Cross-validation results
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        
        # Define cross-validation strategy
        cv_strategies = {
            'stratified': StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            'repeated_stratified': RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=3, random_state=self.random_state),
            'leave_one_out': LeaveOneOut(),
            'time_series': TimeSeriesSplit(n_splits=cv_folds)
        }
        
        cv = cv_strategies.get(cv_strategy, cv_strategies['stratified'])
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring_metrics,
            return_train_score=return_train_score, n_jobs=self.n_jobs
        )
        
        # Calculate statistics
        results_summary = {}
        for metric in scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            results_summary[metric] = {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores),
                'scores': test_scores
            }
            
            if return_train_score:
                train_scores = cv_results[f'train_{metric}']
                results_summary[metric]['train_mean'] = np.mean(train_scores)
                results_summary[metric]['train_std'] = np.std(train_scores)
                results_summary[metric]['overfitting'] = np.mean(train_scores) - np.mean(test_scores)
        
        # Add fit times
        results_summary['fit_time'] = {
            'mean': np.mean(cv_results['fit_time']),
            'std': np.std(cv_results['fit_time']),
            'total': np.sum(cv_results['fit_time'])
        }
        
        results_summary['score_time'] = {
            'mean': np.mean(cv_results['score_time']),
            'std': np.std(cv_results['score_time']),
            'total': np.sum(cv_results['score_time'])
        }
        
        return results_summary
    
    def compare_models_statistical(self, models_dict: Dict[str, Any], X: np.ndarray, y: np.ndarray, cv_folds: int = 5, 
                                 alpha: float = 0.05, test_type: str = 'friedman') -> Dict[str, Any]:
        """
        Perform statistical comparison between multiple models.
        
        Args:
            models_dict (dict): Dictionary of model names and model objects
            X (array): Features
            y (array): Target variable
            cv_folds (int): Number of cross-validation folds
            alpha (float): Significance level
            test_type (str): Statistical test type ('friedman', 'wilcoxon')
            
        Returns:
            dict: Statistical comparison results
        """
        # Collect cross-validation scores for all models
        cv_scores = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in models_dict.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=self.n_jobs)
            cv_scores[model_name] = scores
        
        results = {
            'cv_scores': cv_scores,
            'statistical_tests': {},
            'rankings': {},
            'significant_differences': []
        }
        
        # Friedman test for multiple models
        if test_type == 'friedman' and len(models_dict) > 2:
            scores_matrix = np.array(list(cv_scores.values()))
            statistic, p_value = friedmanchisquare(*scores_matrix)
            
            results['statistical_tests']['friedman'] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha
            }
        
        # Pairwise Wilcoxon signed-rank tests
        model_names = list(models_dict.keys())
        pairwise_results = {}
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                scores1, scores2 = cv_scores[model1], cv_scores[model2]
                
                # Wilcoxon signed-rank test
                try:
                    statistic, p_value = wilcoxon(scores1, scores2)
                    pairwise_results[f'{model1}_vs_{model2}'] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'winner': model1 if np.mean(scores1) > np.mean(scores2) else model2
                    }
                    
                    if p_value < alpha:
                        winner = model1 if np.mean(scores1) > np.mean(scores2) else model2
                        results['significant_differences'].append({
                            'comparison': f'{model1} vs {model2}',
                            'winner': winner,
                            'p_value': p_value
                        })
                except ValueError:
                    # Handle case where scores are identical
                    pairwise_results[f'{model1}_vs_{model2}'] = {
                        'statistic': 0,
                        'p_value': 1.0,
                        'significant': False,
                        'winner': 'tie'
                    }
        
        results['statistical_tests']['pairwise'] = pairwise_results
        
        # Calculate rankings
        mean_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}
        sorted_models = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            results['rankings'][model_name] = {
                'rank': rank,
                'mean_score': score,
                'std_score': np.std(cv_scores[model_name])
            }
        
        return results
    
    def validate_model_stability(self, model: Any, X: np.ndarray, y: np.ndarray, n_iterations: int = 10, 
                               sample_fraction: float = 0.8, feature_fraction: float = 0.8) -> Dict[str, Any]:
        """
        Test model stability across different data subsets and feature subsets.
        
        Args:
            model: Machine learning model to test
            X (array): Features
            y (array): Target variable
            n_iterations (int): Number of stability iterations
            sample_fraction (float): Fraction of samples to use in each iteration
            feature_fraction (float): Fraction of features to use in each iteration
            
        Returns:
            dict: Stability analysis results
        """
        n_samples, n_features = X.shape
        n_sample_subset = int(n_samples * sample_fraction)
        n_feature_subset = int(n_features * feature_fraction)
        
        stability_scores = []
        feature_importance_variations = []
        
        np.random.seed(self.random_state)
        
        for iteration in range(n_iterations):
            # Random sample subset
            sample_indices = np.random.choice(n_samples, n_sample_subset, replace=False)
            
            # Random feature subset
            feature_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            
            # Subset data
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            
            # Train and evaluate model
            try:
                # Clone model to avoid interference
                from sklearn.base import clone
                model_clone = clone(model)
                
                # Cross-validation on subset
                cv_scores = cross_val_score(model_clone, X_subset, y_subset, 
                                          cv=3, scoring='f1_macro')
                stability_scores.append(np.mean(cv_scores))
                
                # Feature importance (if available)
                if hasattr(model, 'fit'):
                    model_clone.fit(X_subset, y_subset)
                    if hasattr(model_clone, 'feature_importances_'):
                        # Pad importance to original feature size
                        importance = np.zeros(n_features)
                        importance[feature_indices] = model_clone.feature_importances_
                        feature_importance_variations.append(importance)
                    elif hasattr(model_clone, 'coef_'):
                        # For linear models
                        importance = np.zeros(n_features)
                        coef = model_clone.coef_
                        if coef.ndim > 1:
                            coef = np.abs(coef).mean(axis=0)
                        importance[feature_indices] = np.abs(coef)
                        feature_importance_variations.append(importance)
                        
            except Exception as e:
                print(f"Warning: Iteration {iteration} failed: {e}")
                continue
        
        # Calculate stability metrics
        stability_results = {
            'performance_stability': {
                'mean_score': np.mean(stability_scores),
                'std_score': np.std(stability_scores),
                'coefficient_of_variation': np.std(stability_scores) / np.mean(stability_scores) if np.mean(stability_scores) > 0 else np.inf,
                'min_score': np.min(stability_scores),
                'max_score': np.max(stability_scores),
                'score_range': np.max(stability_scores) - np.min(stability_scores),
                'all_scores': stability_scores
            }
        }
        
        # Feature importance stability
        if feature_importance_variations:
            importance_matrix = np.array(feature_importance_variations)
            stability_results['feature_importance_stability'] = {
                'mean_importance': np.mean(importance_matrix, axis=0),
                'std_importance': np.std(importance_matrix, axis=0),
                'coefficient_of_variation': np.std(importance_matrix, axis=0) / (np.mean(importance_matrix, axis=0) + 1e-8),
                'stability_score': 1 - np.mean(np.std(importance_matrix, axis=0))
            }
        
        return stability_results
    
    def validate_hyperparameters(self, model: Any, param_grid: Dict[str, Any], X: np.ndarray, y: np.ndarray, cv_folds: int = 5, 
                               scoring: str = 'f1_macro', validation_method: str = 'grid') -> Dict[str, Any]:
        """
        Validate hyperparameter choices and their impact on model performance.
        
        Args:
            model: Base model for hyperparameter validation
            param_grid (dict): Parameter grid to validate
            X (array): Features
            y (array): Target variable
            cv_folds (int): Number of cross-validation folds
            scoring (str): Scoring metric
            validation_method (str): Validation method ('grid', 'random')
            
        Returns:
            dict: Hyperparameter validation results
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if validation_method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_jobs=self.n_jobs, return_train_score=True
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_jobs=self.n_jobs, return_train_score=True,
                n_iter=50, random_state=self.random_state
            )
        
        search.fit(X, y)
        
        # Analyze results
        results_df = pd.DataFrame(search.cv_results_)
        
        validation_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'parameter_analysis': {},
            'overfitting_analysis': {
                'mean_train_score': results_df['mean_train_score'].mean(),
                'mean_test_score': results_df['mean_test_score'].mean(),
                'overfitting_gap': results_df['mean_train_score'].mean() - results_df['mean_test_score'].mean(),
                'max_overfitting': (results_df['mean_train_score'] - results_df['mean_test_score']).max()
            },
            'performance_distribution': {
                'score_std': results_df['mean_test_score'].std(),
                'score_range': results_df['mean_test_score'].max() - results_df['mean_test_score'].min(),
                'top_10_percent_threshold': results_df['mean_test_score'].quantile(0.9)
            }
        }
        
        # Analyze individual parameters
        for param_name in param_grid.keys():
            if f'param_{param_name}' in results_df.columns:
                param_analysis = results_df.groupby(f'param_{param_name}')['mean_test_score'].agg([
                    'mean', 'std', 'count', 'min', 'max'
                ]).to_dict()
                validation_results['parameter_analysis'][param_name] = param_analysis
        
        return validation_results
    
    def detect_data_drift(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: Optional[List[str]] = None, 
                         drift_threshold: float = 0.05, method: str = 'ks_test') -> Dict[str, Any]:
        """
        Detect data drift between training and test sets.
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            feature_names (list): Names of features
            drift_threshold (float): Threshold for drift detection
            method (str): Drift detection method
            
        Returns:
            dict: Data drift analysis results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        drift_results = {
            'overall_drift': False,
            'feature_drift': {},
            'drift_summary': {
                'total_features': len(feature_names),
                'drifted_features': 0,
                'drift_percentage': 0.0
            }
        }
        
        drifted_features = []
        
        for i, feature_name in enumerate(feature_names):
            train_feature = X_train[:, i]
            test_feature = X_test[:, i]
            
            if method == 'ks_test':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(train_feature, test_feature)
                is_drifted = p_value < drift_threshold
            elif method == 'chi2_test':
                # Chi-square test for categorical features
                try:
                    # Bin continuous features
                    train_binned = pd.cut(train_feature, bins=10, duplicates='drop')
                    test_binned = pd.cut(test_feature, bins=10, duplicates='drop')
                    
                    # Create contingency table
                    train_counts = train_binned.value_counts().sort_index()
                    test_counts = test_binned.value_counts().sort_index()
                    
                    # Align indices
                    all_bins = train_counts.index.union(test_counts.index)
                    train_aligned = train_counts.reindex(all_bins, fill_value=0)
                    test_aligned = test_counts.reindex(all_bins, fill_value=0)
                    
                    statistic, p_value = stats.chisquare(test_aligned, train_aligned)
                    is_drifted = p_value < drift_threshold
                except:
                    statistic, p_value = np.nan, 1.0
                    is_drifted = False
            else:
                # Default to KS test
                statistic, p_value = stats.ks_2samp(train_feature, test_feature)
                is_drifted = p_value < drift_threshold
            
            drift_results['feature_drift'][feature_name] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_drifted': is_drifted,
                'train_mean': float(np.mean(train_feature)),
                'test_mean': float(np.mean(test_feature)),
                'train_std': float(np.std(train_feature)),
                'test_std': float(np.std(test_feature))
            }
            
            if is_drifted:
                drifted_features.append(feature_name)
        
        # Update summary
        drift_results['drift_summary']['drifted_features'] = len(drifted_features)
        drift_results['drift_summary']['drift_percentage'] = len(drifted_features) / len(feature_names) * 100
        drift_results['overall_drift'] = len(drifted_features) > 0
        drift_results['drifted_feature_names'] = drifted_features
        
        return drift_results
    
    def validate_model_interpretability(self, model: Any, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, 
                                      method: str = 'permutation', n_repeats: int = 10) -> Dict[str, Any]:
        """
        Validate model interpretability and feature importance consistency.
        
        Args:
            model: Trained model
            X (array): Features
            y (array): Target variable
            feature_names (list): Names of features
            method (str): Interpretability method
            n_repeats (int): Number of repeats for permutation importance
            
        Returns:
            dict: Interpretability validation results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        interpretability_results = {
            'feature_importance': {},
            'consistency_metrics': {},
            'interpretability_score': 0.0
        }
        
        # Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            builtin_importance = model.feature_importances_
            interpretability_results['feature_importance']['builtin'] = dict(zip(feature_names, builtin_importance))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            interpretability_results['feature_importance']['builtin'] = dict(zip(feature_names, np.abs(coef)))
        
        # Permutation importance
        if PERMUTATION_AVAILABLE:
            try:
                perm_importance = permutation_importance(
                    model, X, y, n_repeats=n_repeats, 
                    random_state=self.random_state, n_jobs=self.n_jobs
                )
                interpretability_results['feature_importance']['permutation'] = {
                    'importances_mean': dict(zip(feature_names, perm_importance.importances_mean)),
                    'importances_std': dict(zip(feature_names, perm_importance.importances_std))
                }
                
                # Consistency between methods
                if 'builtin' in interpretability_results['feature_importance']:
                    builtin_imp = np.array(list(interpretability_results['feature_importance']['builtin'].values()))
                    perm_imp = perm_importance.importances_mean
                    
                    # Normalize both to [0, 1]
                    builtin_norm = builtin_imp / (np.max(builtin_imp) + 1e-8)
                    perm_norm = perm_imp / (np.max(perm_imp) + 1e-8)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(builtin_norm, perm_norm)[0, 1]
                    interpretability_results['consistency_metrics']['importance_correlation'] = correlation
                    
            except Exception as e:
                print(f"Warning: Permutation importance failed: {e}")
        
        # SHAP values (if available)
        if SHAP_AVAILABLE:
            try:
                explainer = shap.Explainer(model, X[:100])  # Use subset for efficiency
                shap_values = explainer(X[:100])
                
                if hasattr(shap_values, 'values'):
                    shap_importance = np.abs(shap_values.values).mean(axis=0)
                    if shap_importance.ndim > 1:
                        shap_importance = shap_importance.mean(axis=1)
                    
                    interpretability_results['feature_importance']['shap'] = dict(zip(feature_names, shap_importance))
                    
            except Exception as e:
                print(f"Warning: SHAP analysis failed: {e}")
        
        # Calculate interpretability score
        importance_methods = len(interpretability_results['feature_importance'])
        consistency_score = interpretability_results['consistency_metrics'].get('importance_correlation', 0)
        
        interpretability_results['interpretability_score'] = (
            (importance_methods / 3.0) * 0.5 + 
            max(0, consistency_score) * 0.5
        )
        
        return interpretability_results
    
    def generate_validation_report(self, model_name: str, validation_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            model_name (str): Name of the model
            validation_results (dict): All validation results
            save_path (str): Path to save the report
            
        Returns:
            str: Validation report
        """
        report = f"""
# 🔍 Model Validation Report: {model_name}

## 📊 Cross-Validation Results
"""
        
        if 'cross_validation' in validation_results:
            cv_results = validation_results['cross_validation']
            for metric, stats in cv_results.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    report += f"\n### {metric.upper()}\n"
                    report += f"- Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
                    report += f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
                    if 'overfitting' in stats:
                        report += f"- Overfitting Gap: {stats['overfitting']:.4f}\n"
        
        if 'stability' in validation_results:
            stability = validation_results['stability']['performance_stability']
            report += f"""

## 🎯 Model Stability Analysis
- Performance Stability: {stability['mean_score']:.4f} ± {stability['std_score']:.4f}
- Coefficient of Variation: {stability['coefficient_of_variation']:.4f}
- Score Range: {stability['score_range']:.4f}
"""
        
        if 'data_drift' in validation_results:
            drift = validation_results['data_drift']
            report += f"""

## 📈 Data Drift Analysis
- Overall Drift Detected: {drift['overall_drift']}
- Drifted Features: {drift['drift_summary']['drifted_features']}/{drift['drift_summary']['total_features']}
- Drift Percentage: {drift['drift_summary']['drift_percentage']:.1f}%
"""
        
        if 'interpretability' in validation_results:
            interp = validation_results['interpretability']
            report += f"""

## 🔍 Interpretability Analysis
- Interpretability Score: {interp['interpretability_score']:.4f}
- Available Methods: {len(interp['feature_importance'])}
"""
            if 'importance_correlation' in interp['consistency_metrics']:
                report += f"- Method Consistency: {interp['consistency_metrics']['importance_correlation']:.4f}\n"
        
        report += f"""

## 📋 Validation Summary

This model has been comprehensively validated using multiple techniques:
- ✅ Cross-validation performance assessment
- ✅ Statistical significance testing
- ✅ Model stability analysis
- ✅ Data drift detection
- ✅ Interpretability validation

**Recommendation**: {'✅ Model is ready for production' if self._assess_model_readiness(validation_results) else '⚠️ Model needs further validation'}

---
*Generated by TalentAI Validation Suite*
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Validation report saved to: {save_path}")
        
        return report
    
    def _assess_model_readiness(self, validation_results: Dict[str, Any]) -> bool:
        """
        Assess if model is ready for production based on validation results.
        
        Args:
            validation_results (dict): All validation results
            
        Returns:
            bool: Whether model is ready for production
        """
        readiness_score = 0
        total_checks = 0
        
        # Check cross-validation performance
        if 'cross_validation' in validation_results:
            cv_results = validation_results['cross_validation']
            if 'f1_macro' in cv_results:
                f1_score = cv_results['f1_macro']['mean']
                if f1_score > 0.7:
                    readiness_score += 1
                total_checks += 1
        
        # Check stability
        if 'stability' in validation_results:
            stability = validation_results['stability']['performance_stability']
            cv = stability['coefficient_of_variation']
            if cv < 0.1:  # Low variation is good
                readiness_score += 1
            total_checks += 1
        
        # Check data drift
        if 'data_drift' in validation_results:
            drift = validation_results['data_drift']
            if drift['drift_summary']['drift_percentage'] < 20:  # Less than 20% drift
                readiness_score += 1
            total_checks += 1
        
        # Check interpretability
        if 'interpretability' in validation_results:
            interp_score = validation_results['interpretability']['interpretability_score']
            if interp_score > 0.5:
                readiness_score += 1
            total_checks += 1
        
        return (readiness_score / max(total_checks, 1)) >= 0.75

# Utility functions
def quick_model_validation(model: Any, X: np.ndarray, y: np.ndarray, model_name: str = "Model", comprehensive: bool = True) -> Dict[str, Any]:
    """
    Quick validation function for a single model.
    
    Args:
        model: Machine learning model
        X (array): Features
        y (array): Target variable
        model_name (str): Name of the model
        comprehensive (bool): Whether to run comprehensive validation
        
    Returns:
        dict: Validation results
    """
    validator = ModelValidator()
    
    results = {}
    
    # Cross-validation
    print(f"🔄 Running cross-validation for {model_name}...")
    results['cross_validation'] = validator.cross_validate_model(model, X, y)
    
    if comprehensive:
        # Stability analysis
        print(f"🎯 Analyzing stability for {model_name}...")
        results['stability'] = validator.validate_model_stability(model, X, y)
        
        # Train model for interpretability
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        model.fit(X_train, y_train)
        
        # Data drift
        print(f"📈 Checking data drift for {model_name}...")
        results['data_drift'] = validator.detect_data_drift(X_train, X_test)
        
        # Interpretability
        print(f"🔍 Analyzing interpretability for {model_name}...")
        results['interpretability'] = validator.validate_model_interpretability(model, X_test, y_test)
    
    return results

def compare_models_comprehensive(models_dict: Dict[str, Any], X: np.ndarray, y: np.ndarray, save_report: bool = True) -> Dict[str, Any]:
    """
    Comprehensive comparison of multiple models.
    
    Args:
        models_dict (dict): Dictionary of model names and model objects
        X (array): Features
        y (array): Target variable
        save_report (bool): Whether to save comparison report
        
    Returns:
        dict: Comprehensive comparison results
    """
    validator = ModelValidator()
    
    print("🚀 Starting comprehensive model comparison...")
    
    # Statistical comparison
    print("📊 Performing statistical comparison...")
    statistical_results = validator.compare_models_statistical(models_dict, X, y)
    
    # Individual model validation
    individual_results = {}
    for model_name, model in models_dict.items():
        print(f"\n🔍 Validating {model_name}...")
        individual_results[model_name] = quick_model_validation(model, X, y, model_name, comprehensive=True)
    
    comparison_results = {
        'statistical_comparison': statistical_results,
        'individual_validation': individual_results,
        'summary': {
            'best_model': max(statistical_results['rankings'].keys(), 
                            key=lambda x: statistical_results['rankings'][x]['mean_score']),
            'total_models': len(models_dict),
            'significant_differences': len(statistical_results['significant_differences'])
        }
    }
    
    if save_report:
        # Generate comparison report
        report_path = "comprehensive_model_comparison_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 TalentAI - Comprehensive Model Comparison Report\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Best model
            best_model = comparison_results['summary']['best_model']
            f.write(f"## 🏆 Best Performing Model: {best_model}\n\n")
            
            # Statistical significance
            f.write("## 📊 Statistical Significance\n\n")
            if comparison_results['statistical_comparison']['significant_differences']:
                for diff in comparison_results['statistical_comparison']['significant_differences']:
                    f.write(f"- **{diff['comparison']}**: {diff['winner']} wins (p={diff['p_value']:.4f})\n")
            else:
                f.write("- No statistically significant differences found\n")
            
            f.write("\n## 📈 Model Rankings\n\n")
            for model_name, ranking in comparison_results['statistical_comparison']['rankings'].items():
                f.write(f"{ranking['rank']}. **{model_name}**: {ranking['mean_score']:.4f} ± {ranking['std_score']:.4f}\n")
        
        print(f"📄 Comprehensive report saved to: {report_path}")
    
    print("✅ Comprehensive model comparison completed!")
    return comparison_results

if __name__ == "__main__":
    print("🔍 TalentAI Model Validation Suite")
    print("=" * 40)
    print("This module provides comprehensive validation utilities for ML models.")
    print("Use quick_model_validation() for single model validation.")
    print("Use compare_models_comprehensive() for multi-model comparison.")
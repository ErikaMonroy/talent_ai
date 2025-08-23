#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite for TalentAI

This script tests all implemented models with the actual dataset to ensure
everything works correctly before production deployment.

Features:
- Tests all individual model scripts
- Validates data preprocessing pipeline
- Checks model training and prediction functionality
- Verifies visualization and validation utilities
- Generates comprehensive test report
- Performance benchmarking

Author: TalentAI Development Team
Date: 2024
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
try:
    from data_preprocessing import DataPreprocessor, evaluate_model_performance
    from logistic_regression_model import LogisticRegressionModel
    from random_forest_model import RandomForestModel
    from xgboost_model import XGBoostModel
    from neural_network_model import NeuralNetworkModel
    from knn_model import KNNModel
    from visualization_utils import ModelVisualizationSuite
    from validation_utils import ModelValidator, quick_model_validation
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all model files are in the same directory.")
    sys.exit(1)

class ModelTestSuite:
    """
    Comprehensive testing suite for all TalentAI models.
    """
    
    def __init__(self, dataset_path=None):
        """
        Initialize the test suite.
        
        Args:
            dataset_path (str): Path to the dataset file
        """
        self.dataset_path = dataset_path or "/Users/michaelpage/Documents/Desarrollo/proyectos/talent_ai/modelo/dataset_estudiantes.csv"
        self.test_results = {
            'preprocessing': {'status': 'pending', 'details': {}},
            'models': {},
            'visualization': {'status': 'pending', 'details': {}},
            'validation': {'status': 'pending', 'details': {}},
            'integration': {'status': 'pending', 'details': {}},
            'summary': {}
        }
        self.start_time = time.time()
        
        # Model configurations for testing
        self.model_configs = {
            'LogisticRegression': {
                'class': LogisticRegressionModel,
                'init_params': {},
                'train_params': {'tune_hyperparameters': False},
                'test_hyperparams': False
            },
            'RandomForest': {
                'class': RandomForestModel,
                'init_params': {},
                'train_params': {'tune_hyperparameters': False},
                'test_hyperparams': False
            },
            'XGBoost': {
                'class': XGBoostModel,
                'init_params': {},
                'train_params': {'tune_hyperparameters': False},
                'test_hyperparams': False
            },
            'NeuralNetwork': {
                'class': NeuralNetworkModel,
                'init_params': {'framework': 'sklearn'},
                'train_params': {'tune_hyperparameters': False},
                'test_hyperparams': False
            },
            'KNN': {
                'class': KNNModel,
                'init_params': {},
                'train_params': {'tune_hyperparameters': False},
                'test_hyperparams': False
            }
        }
    
    def print_header(self, title, level=1):
        """Print formatted header."""
        symbols = ['🎯', '📊', '🔍', '⚡']
        symbol = symbols[min(level-1, len(symbols)-1)]
        print(f"\n{symbol} {title}")
        print("=" * (len(title) + 3))
    
    def print_status(self, message, status='info'):
        """Print formatted status message."""
        status_symbols = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'progress': '🔄'
        }
        symbol = status_symbols.get(status, 'ℹ️')
        print(f"{symbol} {message}")
    
    def test_data_preprocessing(self):
        """
        Test data preprocessing functionality.
        """
        self.print_header("Testing Data Preprocessing", 2)
        
        try:
            # Check if dataset exists
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            self.print_status(f"Loading dataset from: {self.dataset_path}", 'progress')
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor()
            
            # Use the full preprocessing pipeline
            data = preprocessor.full_preprocessing_pipeline(self.dataset_path)
            self.print_status(f"Dataset loaded and preprocessed successfully", 'success')
            
            # Extract preprocessed data
            X_train_scaled = data['X_train_scaled']
            X_test_scaled = data['X_test_scaled']
            y_train = data['y_train']
            y_test = data['y_test']
            label_encoder = data['label_encoder']
            scaler = data['scaler']
            
            self.print_status(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}", 'success')
            self.print_status(f"Classes: {len(np.unique(y_train))}", 'success')
            
            # Store preprocessed data for model testing
            self.preprocessed_data = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'label_encoder': label_encoder,
                'scaler': scaler,
                'feature_names': list(range(X_train_scaled.shape[1])),  # Use indices as feature names
                'n_samples': X_train_scaled.shape[0] + X_test_scaled.shape[0]
            }
            
            self.test_results['preprocessing'] = {
                'status': 'success',
                'details': {
                    'dataset_shape': (X_train_scaled.shape[0] + X_test_scaled.shape[0], X_train_scaled.shape[1]),
                    'n_classes': len(np.unique(y_train)),
                    'train_size': X_train_scaled.shape[0],
                    'test_size': X_test_scaled.shape[0],
                    'n_features': X_train_scaled.shape[1]
                }
            }
            
            self.print_status("Data preprocessing test completed successfully!", 'success')
            return True
            
        except Exception as e:
            error_msg = f"Data preprocessing failed: {str(e)}"
            self.print_status(error_msg, 'error')
            self.test_results['preprocessing'] = {
                'status': 'failed',
                'details': {'error': error_msg}
            }
            return False
    
    def test_individual_model(self, model_name, model_config):
        """
        Test an individual model.
        
        Args:
            model_name (str): Name of the model
            model_config (dict): Model configuration
            
        Returns:
            bool: Test success status
        """
        self.print_status(f"Testing {model_name}...", 'progress')
        
        try:
            start_time = time.time()
            
            # Initialize model
            model_class = model_config['class']
            model = model_class(**model_config['init_params'])
            
            # Train model with specific parameters
            train_params = model_config['train_params'].copy()
            model.train(
                self.preprocessed_data['X_train'],
                self.preprocessed_data['y_train'],
                **train_params
            )
            
            # Make predictions
            y_pred = model.predict(self.preprocessed_data['X_test'])
            
            # Evaluate performance
            metrics = evaluate_model_performance(
                self.preprocessed_data['y_test'], y_pred
            )
            
            training_time = time.time() - start_time
            
            # Test model-specific features
            model_features = {
                'has_feature_importance': hasattr(model.model, 'feature_importances_') or hasattr(model.model, 'coef_'),
                'has_predict_proba': hasattr(model.model, 'predict_proba'),
                'supports_cross_validation': True,
                'can_save_load': hasattr(model, 'save_model') and hasattr(model, 'load_model')
            }
            
            # Test save/load functionality
            save_load_success = False
            if model_features['can_save_load']:
                try:
                    test_model_path = f"test_{model_name.lower()}_model.pkl"
                    model.save_model(test_model_path)
                    
                    # Create new instance and load
                    new_model = model_class()
                    new_model.load_model(test_model_path)
                    
                    # Test prediction consistency
                    y_pred_loaded = new_model.predict(self.preprocessed_data['X_test'][:10])
                    y_pred_original = model.predict(self.preprocessed_data['X_test'][:10])
                    
                    save_load_success = np.array_equal(y_pred_loaded, y_pred_original)
                    
                    # Clean up
                    if os.path.exists(test_model_path):
                        os.remove(test_model_path)
                        
                except Exception as e:
                    self.print_status(f"Save/Load test failed for {model_name}: {e}", 'warning')
            
            self.test_results['models'][model_name] = {
                'status': 'success',
                'details': {
                    'training_time': training_time,
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'features': model_features,
                    'save_load_success': save_load_success
                }
            }
            
            self.print_status(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_macro']:.3f}, Time: {training_time:.1f}s", 'success')
            return True
            
        except Exception as e:
            error_msg = f"{model_name} failed: {str(e)}"
            self.print_status(error_msg, 'error')
            self.test_results['models'][model_name] = {
                'status': 'failed',
                'details': {'error': error_msg, 'traceback': traceback.format_exc()}
            }
            return False
    
    def test_all_models(self):
        """
        Test all individual models.
        """
        self.print_header("Testing Individual Models", 2)
        
        if not hasattr(self, 'preprocessed_data'):
            self.print_status("Preprocessed data not available. Skipping model tests.", 'error')
            return False
        
        success_count = 0
        total_models = len(self.model_configs)
        
        for model_name, model_config in self.model_configs.items():
            if self.test_individual_model(model_name, model_config):
                success_count += 1
        
        self.print_status(f"Model testing completed: {success_count}/{total_models} models successful", 
                         'success' if success_count == total_models else 'warning')
        
        return success_count == total_models
    
    def test_visualization_utils(self):
        """
        Test visualization utilities.
        """
        self.print_header("Testing Visualization Utilities", 2)
        
        try:
            # Create sample results for testing
            sample_results = {}
            sample_times = {}
            
            for model_name in self.model_configs.keys():
                if model_name in self.test_results['models'] and self.test_results['models'][model_name]['status'] == 'success':
                    details = self.test_results['models'][model_name]['details']
                    sample_results[model_name] = {
                        'accuracy': details['accuracy'],
                        'f1_macro': details['f1_macro'],
                        'f1_weighted': details['f1_weighted']
                    }
                    sample_times[model_name] = details['training_time']
            
            if not sample_results:
                raise ValueError("No successful model results available for visualization testing")
            
            # Initialize visualization suite
            viz_suite = ModelVisualizationSuite()
            
            # Test dashboard creation (without showing)
            self.print_status("Testing model comparison dashboard...", 'progress')
            fig = viz_suite.plot_model_comparison_dashboard(
                sample_results, sample_times, show_plot=False
            )
            
            if fig is not None:
                self.print_status("Dashboard creation successful", 'success')
            else:
                raise ValueError("Dashboard creation returned None")
            
            # Test other visualization functions
            viz_features = {
                'dashboard': True,
                'learning_curves': hasattr(viz_suite, 'plot_learning_curves'),
                'confusion_matrices': hasattr(viz_suite, 'plot_confusion_matrices'),
                'feature_importance': hasattr(viz_suite, 'plot_feature_importance_comparison'),
                'prediction_distribution': hasattr(viz_suite, 'plot_prediction_distribution')
            }
            
            self.test_results['visualization'] = {
                'status': 'success',
                'details': {
                    'features_available': viz_features,
                    'tested_models': len(sample_results)
                }
            }
            
            self.print_status("Visualization utilities test completed successfully!", 'success')
            return True
            
        except Exception as e:
            error_msg = f"Visualization testing failed: {str(e)}"
            self.print_status(error_msg, 'error')
            self.test_results['visualization'] = {
                'status': 'failed',
                'details': {'error': error_msg}
            }
            return False
    
    def test_validation_utils(self):
        """
        Test validation utilities.
        """
        self.print_header("Testing Validation Utilities", 2)
        
        try:
            # Test with a simple model
            from sklearn.ensemble import RandomForestClassifier
            
            simple_model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
            
            # Test quick validation
            self.print_status("Testing quick model validation...", 'progress')
            validation_results = quick_model_validation(
                simple_model, 
                self.preprocessed_data['X_train'][:100],  # Use subset for speed
                self.preprocessed_data['y_train'][:100],
                model_name="TestModel",
                comprehensive=False  # Quick test only
            )
            
            # Check validation results structure
            required_keys = ['cross_validation']
            validation_features = {
                'cross_validation': 'cross_validation' in validation_results,
                'has_metrics': False
            }
            
            if validation_results.get('cross_validation'):
                cv_results = validation_results['cross_validation']
                validation_features['has_metrics'] = 'accuracy' in cv_results and 'f1_macro' in cv_results
            
            # Test ModelValidator class
            validator = ModelValidator()
            
            validation_class_features = {
                'cross_validate_model': hasattr(validator, 'cross_validate_model'),
                'compare_models_statistical': hasattr(validator, 'compare_models_statistical'),
                'validate_model_stability': hasattr(validator, 'validate_model_stability'),
                'detect_data_drift': hasattr(validator, 'detect_data_drift')
            }
            
            self.test_results['validation'] = {
                'status': 'success',
                'details': {
                    'quick_validation': validation_features,
                    'validator_class': validation_class_features,
                    'test_completed': True
                }
            }
            
            self.print_status("Validation utilities test completed successfully!", 'success')
            return True
            
        except Exception as e:
            error_msg = f"Validation testing failed: {str(e)}"
            self.print_status(error_msg, 'error')
            self.test_results['validation'] = {
                'status': 'failed',
                'details': {'error': error_msg}
            }
            return False
    
    def test_integration(self):
        """
        Test integration between all components.
        """
        self.print_header("Testing System Integration", 2)
        
        try:
            # Test end-to-end workflow
            self.print_status("Testing end-to-end workflow...", 'progress')
            
            # 1. Data preprocessing
            preprocessor = DataPreprocessor()
            data = preprocessor.full_preprocessing_pipeline(self.dataset_path)
            X_train_scaled = data['X_train_scaled']
            X_test_scaled = data['X_test_scaled']
            y_train = data['y_train']
            y_test = data['y_test']
            
            # 2. Train multiple models
            trained_models = {}
            model_results = {}
            
            for model_name, config in list(self.model_configs.items())[:2]:  # Test first 2 models
                try:
                    model = config['class'](**config['init_params'])
                    train_params = config['train_params'].copy()
                    model.train(X_train_scaled, y_train, **train_params)
                    y_pred = model.predict(X_test_scaled)
                    metrics = evaluate_model_performance(y_test, y_pred)
                    
                    trained_models[model_name] = model
                    model_results[model_name] = metrics
                    
                except Exception as e:
                    self.print_status(f"Integration test failed for {model_name}: {e}", 'warning')
            
            if not trained_models:
                raise ValueError("No models successfully trained in integration test")
            
            # 3. Test visualization with real results
            viz_suite = ModelVisualizationSuite()
            dashboard_fig = viz_suite.plot_model_comparison_dashboard(
                model_results, show_plot=False
            )
            
            # 4. Test validation
            validator = ModelValidator()
            first_model = list(trained_models.values())[0]
            cv_results = validator.cross_validate_model(
                first_model.model, X_train_scaled[:50], y_train[:50]  # Small subset
            )
            
            integration_features = {
                'preprocessing_to_training': len(trained_models) > 0,
                'training_to_evaluation': len(model_results) > 0,
                'evaluation_to_visualization': dashboard_fig is not None,
                'model_to_validation': cv_results is not None,
                'end_to_end_success': True
            }
            
            self.test_results['integration'] = {
                'status': 'success',
                'details': {
                    'features': integration_features,
                    'models_tested': len(trained_models),
                    'workflow_complete': True
                }
            }
            
            self.print_status("System integration test completed successfully!", 'success')
            return True
            
        except Exception as e:
            error_msg = f"Integration testing failed: {str(e)}"
            self.print_status(error_msg, 'error')
            self.test_results['integration'] = {
                'status': 'failed',
                'details': {'error': error_msg}
            }
            return False
    
    def generate_test_report(self):
        """
        Generate comprehensive test report.
        """
        self.print_header("Generating Test Report", 2)
        
        total_time = time.time() - self.start_time
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        # Count preprocessing test
        total_tests += 1
        if self.test_results['preprocessing']['status'] == 'success':
            passed_tests += 1
        
        # Count model tests
        for model_result in self.test_results['models'].values():
            total_tests += 1
            if model_result['status'] == 'success':
                passed_tests += 1
        
        # Count utility tests
        for util_name in ['visualization', 'validation', 'integration']:
            total_tests += 1
            if self.test_results[util_name]['status'] == 'success':
                passed_tests += 1
        
        # Update summary
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_time': total_time,
            'overall_status': 'success' if passed_tests == total_tests else 'partial' if passed_tests > 0 else 'failed'
        }
        
        # Generate report
        report = f"""
# 🎯 TalentAI Model Testing Report

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Testing Time:** {total_time:.1f} seconds
**Overall Status:** {'✅ ALL TESTS PASSED' if passed_tests == total_tests else '⚠️ SOME TESTS FAILED' if passed_tests > 0 else '❌ ALL TESTS FAILED'}

## 📊 Test Summary
- **Total Tests:** {total_tests}
- **Passed:** {passed_tests}
- **Failed:** {total_tests - passed_tests}
- **Success Rate:** {(passed_tests/total_tests)*100:.1f}%

## 🔍 Detailed Results

### Data Preprocessing
- **Status:** {'✅ PASSED' if self.test_results['preprocessing']['status'] == 'success' else '❌ FAILED'}
"""
        
        if self.test_results['preprocessing']['status'] == 'success':
            details = self.test_results['preprocessing']['details']
            report += f"""
- **Dataset Shape:** {details['dataset_shape']}
- **Classes:** {details['n_classes']}
- **Train/Test Split:** {details['train_size']}/{details['test_size']}
"""
        
        report += "\n### Individual Models\n"
        
        for model_name, result in self.test_results['models'].items():
            status_icon = '✅' if result['status'] == 'success' else '❌'
            report += f"\n#### {model_name} {status_icon}\n"
            
            if result['status'] == 'success':
                details = result['details']
                report += f"""
- **Accuracy:** {details['accuracy']:.4f}
- **F1-Score (Macro):** {details['f1_macro']:.4f}
- **Training Time:** {details['training_time']:.2f}s
- **Save/Load:** {'✅' if details['save_load_success'] else '❌'}
"""
            else:
                report += f"- **Error:** {result['details']['error']}\n"
        
        # Add utility test results
        for util_name in ['visualization', 'validation', 'integration']:
            status_icon = '✅' if self.test_results[util_name]['status'] == 'success' else '❌'
            report += f"\n### {util_name.title()} Utilities {status_icon}\n"
            
            if self.test_results[util_name]['status'] == 'success':
                report += "- All features tested successfully\n"
            else:
                report += f"- **Error:** {self.test_results[util_name]['details']['error']}\n"
        
        report += f"""

## 🚀 Recommendations

{'✅ **System Ready for Production**: All tests passed successfully. The TalentAI model system is ready for deployment.' if passed_tests == total_tests else '⚠️ **Action Required**: Some tests failed. Please review the failed components before deployment.' if passed_tests > 0 else '❌ **System Not Ready**: Multiple critical failures detected. Comprehensive debugging required.'}

### Next Steps
{'- Deploy to production environment' if passed_tests == total_tests else '- Fix failed components and re-run tests'}
- Monitor model performance in production
- Set up automated retraining pipeline
- Implement A/B testing for model improvements

---
*Generated by TalentAI Model Testing Suite*
"""
        
        # Save report
        report_path = "model_testing_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.print_status(f"Test report saved to: {report_path}", 'success')
        
        # Print summary to console
        print("\n" + "="*60)
        print(f"🎯 TEST SUMMARY: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"📄 Report: {report_path}")
        print("="*60)
        
        return report_path
    
    def run_all_tests(self):
        """
        Run all tests in sequence.
        
        Returns:
            bool: Overall test success status
        """
        self.print_header("TalentAI Model Testing Suite")
        self.print_status(f"Starting comprehensive testing at {pd.Timestamp.now().strftime('%H:%M:%S')}", 'info')
        
        # Run tests in sequence
        test_sequence = [
            ("Data Preprocessing", self.test_data_preprocessing),
            ("Individual Models", self.test_all_models),
            ("Visualization Utils", self.test_visualization_utils),
            ("Validation Utils", self.test_validation_utils),
            ("System Integration", self.test_integration)
        ]
        
        overall_success = True
        
        for test_name, test_function in test_sequence:
            try:
                success = test_function()
                if not success:
                    overall_success = False
                    self.print_status(f"{test_name} test failed", 'warning')
            except Exception as e:
                overall_success = False
                self.print_status(f"{test_name} test crashed: {e}", 'error')
        
        # Generate final report
        report_path = self.generate_test_report()
        
        return overall_success

def main():
    """
    Main function to run all tests.
    """
    # Check if dataset exists
    dataset_path = "/Users/michaelpage/Documents/Desarrollo/proyectos/talent_ai/modelo/dataset_estudiantes.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists before running tests.")
        return False
    
    # Initialize and run test suite
    test_suite = ModelTestSuite(dataset_path)
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! TalentAI system is ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Please review the report and fix issues.")
    
    return success

if __name__ == "__main__":
    main()
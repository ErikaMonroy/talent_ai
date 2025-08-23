#!/usr/bin/env python3
"""
Neural Network Model for TalentAI Project

This module provides a comprehensive neural network implementation supporting both
scikit-learn MLPClassifier and TensorFlow/Keras models for student performance prediction.

Features:
- Dual framework support (scikit-learn and TensorFlow)
- Hyperparameter tuning with GridSearchCV and Optuna
- Advanced architectures with batch normalization and dropout
- GPU support for TensorFlow models
- Comprehensive evaluation and visualization tools
- Model persistence and loading capabilities
- Cross-validation support

"""

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor, evaluate_model_performance
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
warnings.filterwarnings('ignore')

# Optional TensorFlow/Keras imports for advanced features
TENSORFLOW_AVAILABLE = False

try:
    import tensorflow  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, callbacks, optimizers  # type: ignore
    from tensorflow.keras.utils import plot_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
    print(f"TensorFlow {tensorflow.__version__} available")
except ImportError:
    print("TensorFlow not available. Using scikit-learn MLPClassifier only.")
    # Create type-safe placeholder variables
    tensorflow = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    callbacks = None  # type: ignore
    optimizers = None  # type: ignore
    plot_model = None  # type: ignore

class NeuralNetworkModel:
    """
    Neural Network Model with support for both scikit-learn and TensorFlow frameworks.
    
    This class provides a unified interface for training neural networks using either
    scikit-learn's MLPClassifier or TensorFlow/Keras models.
    """
    
    def __init__(self, framework: str = 'sklearn', random_state: int = 42, use_gpu: bool = False):
        """
        Initialize the Neural Network Model.
        
        Args:
            framework (str): Framework to use ('sklearn' or 'tensorflow')
            random_state (int): Random seed for reproducibility
            use_gpu (bool): Whether to use GPU for TensorFlow (if available)
        """
        self.framework = framework.lower()
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        # Model objects
        self.model: Optional[MLPClassifier] = None
        self.keras_model: Optional[Any] = None
        
        # Training state
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.n_classes: Optional[int] = None
        
        # Training history and metrics
        self.training_history: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}
        
        # Validate framework
        if self.framework not in ['sklearn', 'tensorflow']:
            raise ValueError("Framework must be 'sklearn' or 'tensorflow'")
        
        if self.framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available. Use framework='sklearn'")
        
        # Configure GPU if using TensorFlow
        if self.framework == 'tensorflow' and self.use_gpu:
            self._configure_gpu()
    
    def _configure_gpu(self) -> None:
        """
        Configure GPU settings for TensorFlow.
        """
        if TENSORFLOW_AVAILABLE and tensorflow is not None:
            try:
                # Enable memory growth to avoid allocating all GPU memory
                gpus = tensorflow.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tensorflow.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU configuration completed. Available GPUs: {len(gpus)}")
                else:
                    print("No GPUs found. Using CPU.")
                    self.use_gpu = False
            except Exception as e:
                print(f"GPU configuration failed: {e}. Using CPU.")
                self.use_gpu = False
    
    def create_sklearn_model(self, **kwargs) -> MLPClassifier:
        """
        Create a scikit-learn MLPClassifier model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            MLPClassifier: Configured model
        """
        # Default parameters optimized for performance
        default_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'shuffle': True,
            'random_state': self.random_state,
            'tol': 1e-4,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8
        }
        
        # Filter out invalid parameters for MLPClassifier
        sklearn_valid_params = {
            'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size',
            'learning_rate', 'learning_rate_init', 'power_t', 'max_iter', 'shuffle',
            'random_state', 'tol', 'verbose', 'warm_start', 'momentum', 'nesterovs_momentum',
            'early_stopping', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon', 'n_iter_no_change',
            'max_fun'
        }
        
        # Update with user parameters, filtering out invalid ones
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sklearn_valid_params}
        default_params.update(filtered_kwargs)
        
        self.model = MLPClassifier(**default_params)
        return self.model
    
    def create_keras_model(self, input_dim: int, n_classes: int, architecture: str = 'deep', **kwargs) -> Any:
        """
        Create a TensorFlow/Keras model.
        
        Args:
            input_dim (int): Number of input features
            n_classes (int): Number of output classes
            architecture (str): Model architecture type
            **kwargs: Additional parameters
            
        Returns:
            keras.Model: Configured model
        """
        if not TENSORFLOW_AVAILABLE or keras is None:
            raise ValueError("TensorFlow not available. Use framework='sklearn'")
        
        # Import required modules locally to avoid type errors
        from tensorflow import keras as tf_keras  # type: ignore
        from tensorflow.keras import layers as tf_layers  # type: ignore
        from tensorflow.keras import optimizers as tf_optimizers  # type: ignore
        
        # Architecture configurations
        architectures = {
            'simple': [64, 32],
            'medium': [128, 64, 32],
            'deep': [256, 128, 64, 32],
            'wide': [512, 256, 128],
            'custom': kwargs.get('hidden_layers', [128, 64, 32])
        }
        
        hidden_layers = architectures.get(architecture, architectures['deep'])
        
        # Model parameters
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        activation = kwargs.get('activation', 'relu')
        use_batch_norm = kwargs.get('use_batch_norm', True)
        l1_reg = kwargs.get('l1_reg', 0.0)
        l2_reg = kwargs.get('l2_reg', 0.001)
        
        # Build model
        model = tf_keras.Sequential()
        
        # Input layer (without explicit name to avoid conflicts)
        model.add(tf_layers.Dense(hidden_layers[0], 
                              input_dim=input_dim,
                              activation=activation,
                              kernel_regularizer=tf_keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
        
        if use_batch_norm:
            model.add(tf_layers.BatchNormalization())
        
        model.add(tf_layers.Dropout(dropout_rate))
        
        # Hidden layers (without explicit names to avoid conflicts)
        for i, units in enumerate(hidden_layers[1:], 1):
            model.add(tf_layers.Dense(units, 
                                  activation=activation,
                                  kernel_regularizer=tf_keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
            
            if use_batch_norm:
                model.add(tf_layers.BatchNormalization())
            
            model.add(tf_layers.Dropout(dropout_rate))
        
        # Output layer (without explicit name to avoid conflicts)
        model.add(tf_layers.Dense(n_classes, 
                              activation='softmax'))
        
        # Compile model
        optimizer_name = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer_obj = tf_optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer_obj = tf_optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer_obj = tf_optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer_obj = optimizer_name
        
        model.compile(
            optimizer=optimizer_obj,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        self.keras_model = model
        return model
    
    def sklearn_hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                                     cv: int = 5, scoring: str = 'f1_macro') -> GridSearchCV:
        """
        Perform hyperparameter tuning for scikit-learn model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            cv (int): Cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        print("Starting hyperparameter tuning for Neural Network...")
        
        # Optimized parameter grid for faster tuning
        param_grid = {
            'hidden_layer_sizes': [
                (64, 32),
                (128, 64),
                (128, 64, 32),
                (256, 128, 64)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [300, 500]
        }
        
        # Create base model
        base_model = MLPClassifier(
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
              feature_names: Optional[List[str]] = None, 
              tune_hyperparameters: bool = True, **kwargs) -> Union[MLPClassifier, Any]:
        """
        Train the neural network model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            X_val (array, optional): Validation features
            y_val (array, optional): Validation labels
            feature_names (list, optional): Feature names
            tune_hyperparameters (bool): Whether to tune hyperparameters
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        print(f"Training Neural Network with {self.framework} framework...")
        
        # Store metadata
        self.feature_names = feature_names
        self.n_classes = len(np.unique(y_train))
        
        # Train based on framework
        if self.framework == 'sklearn':
            return self._train_sklearn(X_train, y_train, tune_hyperparameters, **kwargs)
        elif self.framework == 'tensorflow':
            return self._train_tensorflow(X_train, y_train, X_val, y_val, **kwargs)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
    
    def _train_sklearn(self, X_train: np.ndarray, y_train: np.ndarray, 
                      tune_hyperparameters: bool = True, **kwargs) -> MLPClassifier:
        """
        Train scikit-learn model.
        """
        if tune_hyperparameters:
            # Perform hyperparameter tuning
            self.sklearn_hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters
            self.create_sklearn_model(**kwargs)
            if self.model is not None:
                self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Cross-validation evaluation
        if self.model is not None:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_macro')
            print(f"Cross-validation F1-macro scores: {cv_scores}")
            print(f"Mean CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            return self.model
        else:
            raise ValueError("Model creation failed")
    
    def _train_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: Optional[np.ndarray] = None, 
                         y_val: Optional[np.ndarray] = None, **kwargs) -> Any:
        """
        Train TensorFlow/Keras model.
        """
        # Training parameters
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        patience = kwargs.get('patience', 10)
        architecture = kwargs.get('architecture', 'deep')
        
        # Create model (clear existing model first to avoid naming conflicts)
        input_dim = X_train.shape[1]
        if self.n_classes is None:
            raise ValueError("Number of classes not determined")
        
        # Clear existing model to prevent layer name conflicts
        self.keras_model = None
        
        # Clear TensorFlow session if needed to prevent naming conflicts
        if TENSORFLOW_AVAILABLE and keras is not None:
            try:
                keras.backend.clear_session()
            except Exception:
                pass  # Ignore if clearing session fails
        
        self.create_keras_model(input_dim, self.n_classes, architecture, **kwargs)
        
        # Prepare callbacks
        callback_list = []
        
        if TENSORFLOW_AVAILABLE:
            # Import callbacks locally
            from tensorflow.keras import callbacks as tf_callbacks  # type: ignore
            
            # Early stopping
            early_stopping = tf_callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stopping)
            
            # Learning rate reduction
            lr_reduction = tf_callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
            callback_list.append(lr_reduction)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        if self.keras_model is not None:
            history = self.keras_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callback_list,
                verbose=1
            )
            
            # Store training history
            self.training_history = history.history
            self.is_trained = True
            
            print(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            if 'val_loss' in history.history:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
            
            return self.keras_model
        else:
            raise ValueError("Keras model creation failed")
    
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
        
        if self.framework == 'sklearn':
            if self.model is not None:
                predictions = self.model.predict(X_test)
                return np.asarray(predictions)
        elif self.framework == 'tensorflow':
            if self.keras_model is not None:
                predictions = self.keras_model.predict(X_test)
                return np.argmax(predictions, axis=1)
        
        raise ValueError("Model not properly initialized")
    
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
        
        if self.framework == 'sklearn':
            if self.model is not None:
                probabilities = self.model.predict_proba(X_test)
                return np.asarray(probabilities)
        elif self.framework == 'tensorflow':
            if self.keras_model is not None:
                probabilities = self.keras_model.predict(X_test)
                return np.asarray(probabilities)
        
        raise ValueError("Model not properly initialized")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Store performance metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        return self.performance_metrics
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot training history for TensorFlow models.
        
        Args:
            figsize (tuple): Figure size
        """
        if self.framework != 'tensorflow' or not self.training_history:
            print("Training history not available for this model.")
            return
        
        history = self.training_history
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot training & validation loss
        axes[0].plot(history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot training & validation accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', color='blue')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        # Plot learning rate (if available)
        if 'lr' in history:
            axes[2].plot(history['lr'], label='Learning Rate', color='green')
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].legend()
            axes[2].grid(True)
        else:
            # Plot loss again if no learning rate data
            axes[2].plot(history['loss'], label='Training Loss', color='blue')
            if 'val_loss' in history:
                axes[2].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[2].set_title('Model Loss (Duplicate)')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_architecture(self, filename: Optional[str] = None, 
                               show_shapes: bool = True, show_layer_names: bool = True) -> None:
        """
        Plot model architecture for TensorFlow models.
        
        Args:
            filename (str, optional): Save plot to file
            show_shapes (bool): Show layer shapes
            show_layer_names (bool): Show layer names
        """
        if self.framework != 'tensorflow' or self.keras_model is None:
            print("Model architecture plotting only available for TensorFlow models.")
            return
        
        if not TENSORFLOW_AVAILABLE or plot_model is None:
            print("TensorFlow plot_model not available.")
            return
        
        try:
            if filename:
                plot_model(self.keras_model, to_file=filename, 
                          show_shapes=show_shapes, show_layer_names=show_layer_names)
                print(f"Model architecture saved to: {filename}")
            else:
                # Display model summary instead
                print("\nModel Architecture Summary:")
                if hasattr(self.keras_model, 'summary'):
                    self.keras_model.summary()
        except Exception as e:
            print(f"Could not plot model architecture: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            dict: Model summary information
        """
        summary = {
            'framework': self.framework,
            'is_trained': self.is_trained,
            'n_classes': self.n_classes,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'random_state': self.random_state
        }
        
        # Add framework-specific information
        if self.framework == 'sklearn' and self.model is not None:
            summary['hidden_layer_sizes'] = getattr(self.model, 'hidden_layer_sizes', None)
            summary['activation'] = getattr(self.model, 'activation', None)
            summary['solver'] = getattr(self.model, 'solver', None)
            summary['n_iter'] = getattr(self.model, 'n_iter_', None)
            summary['loss'] = getattr(self.model, 'loss_', None)
        
        if self.framework == 'tensorflow' and self.keras_model is not None and TENSORFLOW_AVAILABLE:
            try:
                summary['total_params'] = self.keras_model.count_params()
                try:
                    from tensorflow.keras import backend as K  # type: ignore
                    summary['trainable_params'] = sum([K.count_params(w) 
                                                     for w in self.keras_model.trainable_weights])
                except (ImportError, AttributeError):
                    summary['trainable_params'] = 'Unknown'
            except (AttributeError, TypeError):
                summary['total_params'] = 'Unknown'
                summary['trainable_params'] = 'Unknown'
        
        if self.best_params:
            summary['best_params'] = self.best_params
        
        if self.performance_metrics:
            summary['performance_metrics'] = self.performance_metrics
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if self.framework == 'sklearn' and self.model is not None:
            # Save scikit-learn model
            joblib.dump(self.model, filepath)
            print(f"Scikit-learn model saved to: {filepath}")
            
        elif self.framework == 'tensorflow' and self.keras_model is not None:
            # Save Keras model
            keras_filepath = filepath.replace('.joblib', '.h5')
            if hasattr(self.keras_model, 'save'):
                self.keras_model.save(keras_filepath)
                print(f"Keras model saved to: {keras_filepath}")
            
            # Save additional data
            model_data = {
                'framework': self.framework,
                'feature_names': self.feature_names,
                'n_classes': self.n_classes,
                'random_state': self.random_state,
                'use_gpu': self.use_gpu,
                'training_history': self.training_history,
                'best_params': self.best_params,
                'performance_metrics': self.performance_metrics,
                'keras_model_path': keras_filepath
            }
            
            metadata_filepath = filepath.replace('.joblib', '_metadata.joblib')
            joblib.dump(model_data, metadata_filepath)
            print(f"Model metadata saved to: {metadata_filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if self.framework == 'sklearn':
            # Load scikit-learn model
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"Scikit-learn model loaded from: {filepath}")
            
        elif self.framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # Load metadata first
            metadata_filepath = filepath.replace('.joblib', '_metadata.joblib')
            model_data = joblib.load(metadata_filepath)
            
            # Restore attributes
            self.feature_names = model_data['feature_names']
            self.n_classes = model_data['n_classes']
            self.random_state = model_data['random_state']
            self.use_gpu = model_data.get('use_gpu', False)
            self.training_history = model_data['training_history']
            self.performance_metrics = model_data['performance_metrics']
            self.best_params = model_data['best_params']
            
            # Load Keras model
            keras_filepath = model_data['keras_model_path']
            try:
                from tensorflow import keras as tf_keras  # type: ignore
                self.keras_model = tf_keras.models.load_model(keras_filepath)
                print(f"Keras model loaded from: {keras_filepath}")
            except (ImportError, AttributeError) as e:
                print(f"Could not load Keras model: {e}")
            
            self.is_trained = True
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                      scoring: str = 'f1_macro') -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (array): Features
            y (array): Labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        if self.framework == 'sklearn':
            if self.model is None:
                # Create a default model for cross-validation
                self.create_sklearn_model()
            
            if self.model is not None:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
                
                results = {
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scoring_metric': scoring,
                    'cv_folds': cv
                }
                
                print(f"Cross-validation {scoring} scores: {scores}")
                print(f"Mean {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
                return results
            else:
                raise ValueError("Model creation failed")
        else:
            print("Cross-validation currently only supported for scikit-learn framework.")
            return {
                'error': 'Cross-validation not supported for TensorFlow framework'
            }

if __name__ == "__main__":
    # Test the Neural Network Model
    print("Testing Neural Network Model...")
    
    # Generate sample data
    np.random.seed(42)
    X_sample = np.random.randn(1000, 20)
    y_sample = np.random.randint(0, 5, 1000)
    
    # Test scikit-learn framework
    print("\n=== Testing Scikit-learn Framework ===")
    nn_sklearn = NeuralNetworkModel(framework='sklearn')
    nn_sklearn.train(X_sample, y_sample, tune_hyperparameters=False)
    
    predictions = nn_sklearn.predict(X_sample[:100])
    probabilities = nn_sklearn.predict_proba(X_sample[:100])
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test TensorFlow framework (if available)
    if TENSORFLOW_AVAILABLE:
        print("\n=== Testing TensorFlow Framework ===")
        nn_tf = NeuralNetworkModel(framework='tensorflow')
        nn_tf.train(X_sample, y_sample, epochs=10, batch_size=32)
        
        predictions_tf = nn_tf.predict(X_sample[:100])
        probabilities_tf = nn_tf.predict_proba(X_sample[:100])
        
        print(f"TF Predictions shape: {predictions_tf.shape}")
        print(f"TF Probabilities shape: {probabilities_tf.shape}")
    
    print("\nNeural Network Model test completed!")
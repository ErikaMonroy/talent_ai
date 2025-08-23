#!/usr/bin/env python3
"""
Data Preprocessing Utilities for TalentAI Model Training

This module provides shared preprocessing functions for all machine learning models
in the TalentAI project. It handles data loading, feature engineering, scaling,
and train-test splitting with proper stratification.

"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for the TalentAI project.
    
    This class handles all data preprocessing steps including:
    - Data loading and validation
    - Feature and target separation
    - Label encoding for target variables
    - Train-test splitting with stratification
    - Feature scaling using StandardScaler
    """
    
    def __init__(self, dataset_path: Optional[str] = None) -> None:
        """
        Initialize the DataPreprocessor.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: Optional[List[str]] = None
        self.target_column = 'area_conocimiento'
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[Union[pd.Series, np.ndarray]] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None
        
    def load_data(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Args:
            dataset_path (str, optional): Path to dataset. Uses instance path if None.
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if dataset_path:
            self.dataset_path = dataset_path
            
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
            
        try:
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully: {self.data.shape}")
            print(f"Features: {self.data.columns.tolist()}")
            return self.data
        except Exception as e:
            raise FileNotFoundError(f"Error loading dataset: {e}")
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform initial data exploration and validation.
        
        Returns:
            dict: Dictionary containing exploration results
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        exploration = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'null_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'target_distribution': self.data[self.target_column].value_counts().to_dict(),
            'feature_stats': self.data.describe().to_dict()
        }
        
        print("=== DATA EXPLORATION ===")
        print(f"Dataset Shape: {exploration['shape']}")
        print(f"Null Values: {sum(exploration['null_values'].values())}")
        print(f"Target Classes: {len(exploration['target_distribution'])}")
        print(f"Target Range: {min(exploration['target_distribution'].keys())} - {max(exploration['target_distribution'].keys())}")
        
        return exploration
    
    def prepare_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Returns:
            tuple: (X, y) features and target arrays
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        # Define feature columns (all except target)
        self.feature_columns = [col for col in self.data.columns if col != self.target_column]
        
        # Separate features and target
        self.X = self.data[self.feature_columns].copy()  # type: ignore
        self.y = self.data[self.target_column].copy()  # type: ignore
        
        if self.X is not None and self.y is not None:
            print(f"Features prepared: {self.X.shape}")
            print(f"Target prepared: {self.y.shape}")
            print(f"Feature columns: {self.feature_columns}")
        
        return self.X, self.y
    
    def encode_target(self) -> np.ndarray:
        """
        Encode target variable from 1-30 to 0-29 for model compatibility.
        
        Returns:
            np.array: Encoded target variable
        """
        if self.y is None:
            raise ValueError("Target variable must be prepared first")
            
        # Transform target from 1-30 to 0-29
        self.y = self.label_encoder.fit_transform(self.y)  # type: ignore
        
        if self.y is not None:
            unique_vals = np.unique(self.y)
            print(f"Target encoded: {len(unique_vals)} classes (0-{max(unique_vals)})")
        
        return self.y
    
    def split_data(self, test_size: float = 0.3, random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            test_size (float): Proportion of test set (default: 0.3)
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify split by target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and target must be prepared first")
            
        stratify_param = self.y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )
        
        # Type-safe assignments
        self.X_train = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.X_test = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        self.y_train = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
        self.y_test = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
        
        if self.X_train is not None and self.X_test is not None and self.y_train is not None:
            print(f"Data split completed:")
            print(f"  Training set: {self.X_train.shape}")
            print(f"  Test set: {self.X_test.shape}")
            print(f"  Training target distribution: {np.bincount(self.y_train)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler fitted on training data only.
        
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data must be split first")
            
        # Fit scaler on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # type: ignore
        self.X_test_scaled = self.scaler.transform(self.X_test)  # type: ignore
        
        if self.X_train_scaled is not None and self.X_test_scaled is not None:
            print(f"Features scaled:")
            print(f"  Training set scaled: {self.X_train_scaled.shape}")
            print(f"  Test set scaled: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def get_preprocessed_data(self) -> Dict[str, Any]:
        """
        Get all preprocessed data components.
        
        Returns:
            dict: Dictionary containing all preprocessed data
        """
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled,
            'X_test_scaled': self.X_test_scaled,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
    
    def full_preprocessing_pipeline(self, dataset_path: Optional[str] = None, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            dataset_path (str, optional): Path to dataset
            test_size (float): Test set proportion
            random_state (int): Random seed
            
        Returns:
            dict: Complete preprocessed data
        """
        print("=== STARTING FULL PREPROCESSING PIPELINE ===")
        
        # Step 1: Load data
        self.load_data(dataset_path)
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Prepare features and target
        self.prepare_features_target()
        
        # Step 4: Encode target
        self.encode_target()
        
        # Step 5: Split data
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Step 6: Scale features
        self.scale_features()
        
        print("=== PREPROCESSING PIPELINE COMPLETED ===")
        
        return self.get_preprocessed_data()

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Evaluate model performance with standard metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n=== {model_name.upper()} PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': class_report
    }

if __name__ == "__main__":
    # Example usage
    dataset_path = "../dataset_estudiantes.csv"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(dataset_path)
    
    # Run full pipeline
    data = preprocessor.full_preprocessing_pipeline()
    
    print("\nPreprocessing completed successfully!")
    print(f"Ready for model training with {data['X_train_scaled'].shape[0]} training samples")
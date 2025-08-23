import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """Data preprocessing utilities for TalentAI models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'area_conocimiento') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target from dataframe"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        for column in X.select_dtypes(include=['object']).columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        
        # Handle target variable
        if y.dtype == 'object':
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
            y = self.label_encoders['target'].fit_transform(y.astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, np.array(y)
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        X = df.copy()
        
        # Handle categorical variables
        for column in X.select_dtypes(include=['object']).columns:
            if column in self.label_encoders:
                X[column] = self.label_encoders[column].transform(X[column].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return np.array(X_scaled)

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """Evaluate model performance and return metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
    except:
        report = {}
    
    # Generate confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
    except:
        cm = np.array([])
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist() if cm.size > 0 else [],
        'n_samples': len(y_true)
    }
    
    return results

def load_and_preprocess_data(file_path: str, target_column: str = 'area_conocimiento') -> Tuple[np.ndarray, np.ndarray, DataPreprocessor]:
    """Load and preprocess data from CSV file"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare features and target
    X, y = preprocessor.prepare_features(df, target_column)
    
    return X, y, preprocessor
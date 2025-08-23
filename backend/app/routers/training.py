from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import asyncio
import json
from pathlib import Path

from app.database.connection import get_db
from app.core.config import settings
from app.schemas.schemas import (
    TrainingRequestKNN, 
    TrainingRequestNN,
    TrainingResponse, 
    ModelType,
    ModelVersion,
    ModelVersionCreate
)
from app.database.models import ModelVersion as ModelVersionDB
from ml_models.knn_model import KNNModel
from ml_models.neural_network_model import NeuralNetworkModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Global training status tracking
training_status = {}

# Model storage paths
MODEL_STORAGE_PATH = Path(settings.MODELS_DIR)
MODEL_STORAGE_PATH.mkdir(exist_ok=True)

def load_training_data() -> pd.DataFrame:
    """Load training data from CSV file"""
    try:
        data_path = settings.DATA_DIR / "datasets" / "dataset_estudiantes.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded training data with {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def prepare_features_and_target(df: pd.DataFrame) -> tuple:
    """Prepare features and target from dataframe"""
    try:
        # Define feature columns (all except target and metadata)
        feature_columns = [
            'matematicas', 'lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas', 'ingles',
            'dimension_1_logico_matematico', 'dimension_2_comprension_comunicacion', 
            'dimension_3_pensamiento_cientifico', 'dimension_4_analisis_social_humanistico',
            'dimension_5_creatividad_innovacion', 'dimension_6_liderazgo_trabajo_equipo',
            'dimension_7_pensamiento_critico', 'dimension_8_adaptabilidad_aprendizaje'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X = df[feature_columns].values
        y = df['area_conocimiento'].values
        
        # CRITICAL FIX: Encode target labels from 1-30 to 0-29 for TensorFlow compatibility
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        logger.info(f"Target encoded from range [{np.min(np.asarray(y))}-{np.max(np.asarray(y))}] to [{np.min(np.asarray(y_encoded))}-{np.max(np.asarray(y_encoded))}]")
        
        return X, y_encoded, feature_columns
    except Exception as e:
        logger.error(f"Error preparing features and target: {str(e)}")
        raise

def generate_model_version(model_type: str) -> str:
    """Generate a new model version string"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_v{timestamp}"

def save_model_metadata(db: Session, model_data: dict) -> ModelVersionDB:
    """Save model metadata to database"""
    try:
        model_version = ModelVersionDB(
            model_type=model_data['model_type'],
            version=model_data['version'],
            file_path=model_data['file_path'],
            training_accuracy=model_data.get('training_accuracy'),
            validation_accuracy=model_data.get('validation_accuracy'),
            training_samples=model_data.get('training_samples'),
            hyperparameters=model_data.get('hyperparameters'),
            is_active=True
        )
        
        # Deactivate previous versions of the same model type
        db.query(ModelVersionDB).filter(
            ModelVersionDB.model_type == model_data['model_type'],
            ModelVersionDB.is_active == True
        ).update({"is_active": False})
        
        db.add(model_version)
        db.commit()
        db.refresh(model_version)
        
        logger.info(f"Saved model metadata: {model_data['version']}")
        return model_version
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving model metadata: {str(e)}")
        raise

async def train_model_background(model_type: str, hyperparameters: Optional[Dict], validation_split: float, db: Session):
    """Background task for model training"""
    training_id = f"{model_type}_{datetime.now(timezone.utc).isoformat()}"
    
    try:
        # Update training status
        training_status[training_id] = {
            "status": "loading_data",
            "progress": 10,
            "message": "Loading training data...",
            "start_time": datetime.now(timezone.utc)
        }
        
        # Load and prepare data
        df = load_training_data()
        X, y, feature_columns = prepare_features_and_target(df)
        
        training_status[training_id].update({
            "status": "training",
            "progress": 30,
            "message": "Training model..."
        })
        
        # Initialize model based on type
        if model_type == ModelType.KNN:
            model = KNNModel()
        elif model_type == ModelType.NEURAL_NETWORK:
            model = NeuralNetworkModel(framework='tensorflow')  # Use sklearn for consistency
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Convert to numpy arrays with proper dtype
        X_array = np.array(X, dtype=np.float64)
        y_array = np.array(y)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_array, y_array, test_size=validation_split, random_state=42, stratify=y_array
        )
        
        # Ensure arrays are proper numpy arrays
        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        
        # Train model
        if model_type == ModelType.KNN:
            trained_model = model.train(X_train, y_train, tune_hyperparameters=False, find_optimal_k_only=True, plot_results=False)
            # Evaluate on validation set
            val_predictions = model.predict(X_val)
            from sklearn.metrics import accuracy_score
            training_accuracy = model.evaluate(X_train, y_train).get('accuracy', 0.0)
            validation_accuracy = accuracy_score(y_val, val_predictions)
        else:  # Neural Network
            # Pass hyperparameters to the training function with correct parameter names
            train_kwargs: Dict[str, Any] = {"tune_hyperparameters": False}
            if hyperparameters:
                # Filter out invalid parameters for neural network training
                valid_nn_params = {'epochs', 'batch_size', 'patience', 'architecture', 'dropout_rate', 
                                 'activation', 'use_batch_norm', 'l1_reg', 'l2_reg', 'optimizer', 'learning_rate'}
                filtered_hyperparams = {k: v for k, v in hyperparameters.items() if k in valid_nn_params}
                train_kwargs.update(filtered_hyperparams)
            
            # Train with validation data - pass validation data through kwargs
            train_kwargs['X_val'] = X_val
            train_kwargs['y_val'] = y_val
            trained_model = model.train(X_train, y_train, feature_names=None, **train_kwargs)
            
            # Manually split and evaluate since TensorFlow model handles validation internally
            training_metrics = model.evaluate(X_train, y_train)
            validation_metrics = model.evaluate(X_val, y_val)
            training_accuracy = training_metrics.get('accuracy', 0.0)
            validation_accuracy = validation_metrics.get('accuracy', 0.0)
        
        training_status[training_id].update({
            "status": "saving",
            "progress": 80,
            "message": "Saving model..."
        })
        
        # Generate version and save model
        model_type_str = model_type.value if isinstance(model_type, ModelType) else str(model_type)
        model_version = generate_model_version(model_type_str)
        
        # Use appropriate file extension based on model type and framework
        if (model_type == ModelType.NEURAL_NETWORK and 
            isinstance(model, NeuralNetworkModel) and 
            hasattr(model, 'framework') and 
            model.framework == 'tensorflow'):
            model_filename = f"{model_version}_metadata.joblib"
        else:
            model_filename = f"{model_version}.pkl"
            
        model_path = MODEL_STORAGE_PATH / model_filename
        
        # Save model file
        model.save_model(str(model_path))
        
        # Save metadata to database
        model_data = {
             "model_type": model_type,
             "version": model_version,
             "file_path": str(model_path),
             "training_accuracy": float(training_accuracy) if training_accuracy else None,
             "validation_accuracy": float(validation_accuracy) if validation_accuracy else None,
             "training_samples": len(X),
             "hyperparameters": hyperparameters or {}
         }
        
        save_model_metadata(db, model_data)
        
        # Update final status
        training_status[training_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully",
            "model_version": model_version,
            "training_accuracy": float(training_accuracy) if training_accuracy else None,
            "validation_accuracy": float(validation_accuracy) if validation_accuracy else None,
            "end_time": datetime.now(timezone.utc)
        })
        
        logger.info(f"Training completed for {model_type}: {model_version}")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        training_status[training_id].update({
            "status": "failed",
            "progress": 0,
            "message": error_msg,
            "error": str(e),
            "end_time": datetime.now(timezone.utc)
        })

@router.get("/training/models")
async def list_available_models(db: Session = Depends(get_db)):
    """List all available model versions"""
    try:
        models = db.query(ModelVersionDB).order_by(ModelVersionDB.created_at.desc()).all()
        
        model_list = []
        for model in models:
            model_list.append({
                "id": model.id,
                "model_type": model.model_type,
                "version": model.version,
                "training_accuracy": model.training_accuracy,
                "validation_accuracy": model.validation_accuracy,
                "training_samples": model.training_samples,
                "is_active": model.is_active,
                "created_at": model.created_at
            })
        
        return {
            "models": model_list,
            "total": len(model_list)
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )

@router.post("/training/knn", response_model=TrainingResponse)
async def train_knn_model(
    request: TrainingRequestKNN,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Train KNN model"""
    try:
        # Check if training data exists
        data_path = settings.DATA_DIR / "datasets" / "dataset_estudiantes.csv"
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training data not found. Please generate dataset first."
            )
        
        # Check for existing recent training
        if not request.force_retrain:
            recent_model = db.query(ModelVersionDB).filter(
                ModelVersionDB.model_type == ModelType.KNN,
                ModelVersionDB.is_active == True
            ).first()
            
            if recent_model:
                time_diff = datetime.now(timezone.utc) - recent_model.created_at
                if time_diff.total_seconds() < 3600:  # Less than 1 hour
                    return TrainingResponse(
                        status="skipped",
                        message=f"Recent KNN model exists: {getattr(recent_model, 'version', None)}",
                        model_version=getattr(recent_model, 'version', None),
                        training_accuracy=getattr(recent_model, 'training_accuracy', None),
                        validation_accuracy=getattr(recent_model, 'validation_accuracy', None)
                    )
        
        # Start background training
        training_id = f"knn_{datetime.now(timezone.utc).isoformat()}"
        training_status[training_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Training queued...",
            "start_time": datetime.now(timezone.utc)
        }
        
        background_tasks.add_task(
            train_model_background,
            ModelType.KNN.value,
            request.hyperparameters,
            request.validation_split,
            db
        )
        
        return TrainingResponse(
            status="started",
            message="KNN training started in background",
            model_version=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting KNN training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting KNN training: {str(e)}"
        )

@router.post("/training/neural-network", response_model=TrainingResponse)
async def train_neural_network(
    request: TrainingRequestNN,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Train Neural Network model"""
    try:
        # Check if training data exists
        data_path = settings.DATA_DIR / "datasets" / "dataset_estudiantes.csv"
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training data not found. Please generate dataset first."
            )
        
        # Check for existing recent training
        if not request.force_retrain:
            recent_model = db.query(ModelVersionDB).filter(
                ModelVersionDB.model_type == ModelType.NEURAL_NETWORK,
                ModelVersionDB.is_active == True
            ).first()
            
            if recent_model:
                time_diff = datetime.now(timezone.utc) - recent_model.created_at
                if time_diff.total_seconds() < 10:  # Less than 1 hour
                    return TrainingResponse(
                        status="skipped",
                        message=f"Recent Neural Network model exists: {getattr(recent_model, 'version', None)}",
                        model_version=getattr(recent_model, 'version', None),
                        training_accuracy=getattr(recent_model, 'training_accuracy', None),
                        validation_accuracy=getattr(recent_model, 'validation_accuracy', None)
                    )
        
        # Start background training
        training_id = f"neural_network_{datetime.now(timezone.utc).isoformat()}"
        training_status[training_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Training queued...",
            "start_time": datetime.now()
        }
        
        background_tasks.add_task(
            train_model_background,
            ModelType.NEURAL_NETWORK.value,
            request.hyperparameters,
            request.validation_split,
            db
        )
        
        return TrainingResponse(
            status="started",
            message="Neural Network training started in background",
            model_version=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting Neural Network training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting Neural Network training: {str(e)}"
        )

@router.get("/training/status/{model_type}")
async def get_training_status(model_type: str, db: Session = Depends(get_db)):
    """Get training status for a specific model type"""
    try:
        # Find active training sessions for this model type
        active_trainings = {
            k: v for k, v in training_status.items() 
            if k.startswith(model_type.lower()) and v.get("status") not in ["completed", "failed"]
        }
        
        # Get latest completed model
        latest_model = db.query(ModelVersionDB).filter(
            ModelVersionDB.model_type == model_type,
            ModelVersionDB.is_active == True
        ).first()
        
        response = {
            "model_type": model_type,
            "active_trainings": len(active_trainings),
            "training_sessions": active_trainings,
            "latest_model": None
        }
        
        if latest_model:
            response["latest_model"] = {
                "version": latest_model.version,
                "training_accuracy": latest_model.training_accuracy,
                "validation_accuracy": latest_model.validation_accuracy,
                "created_at": latest_model.created_at
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting training status: {str(e)}"
        )
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pickle
from pathlib import Path
import json

from app.database.connection import get_db
from app.core.config import settings
from app.schemas.schemas import (
    PredictionInput, 
    PredictionResponse, 
    AreaPrediction,
    ModelType
)
from app.database.models import ModelVersion as ModelVersionDB, Prediction
from ml_models.knn_model import KNNModel
from ml_models.neural_network_model import NeuralNetworkModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for loaded models
model_cache = {}

# Mapeo de nombres de áreas a IDs basado en areas_conocimiento.csv
AREA_NAME_TO_ID = {
    'Administración y Gestión Empresarial': 1,
    'Finanzas y Contabilidad': 2,
    'Mercadeo y Ventas': 3,
    'Turismo y Hotelería': 4,
    'Gastronomía y Cocina': 5,
    'Belleza y Estética': 6,
    'Atención al Cliente y Servicios': 7,
    'Sistemas e Informática': 8,
    'Redes y Telecomunicaciones': 9,
    'Ingeniería Civil y Construcción': 10,
    'Ingeniería Industrial y Procesos': 11,
    'Electrónica y Automatización': 12,
    'Enfermería y Auxiliares de Salud': 13,
    'Salud Pública y Comunitaria': 14,
    'Farmacia y Servicios Farmacéuticos': 15,
    'Educación y Pedagogía': 16,
    'Primera Infancia y Cuidado Infantil': 17,
    'Psicología y Trabajo Social': 18,
    'Seguridad y Protección': 19,
    'Arte y Cultura': 20,
    'Música y Artes Escénicas': 21,
    'Diseño Gráfico y Multimedia': 22,
    'Agricultura y Ganadería': 23,
    'Medio Ambiente y Sostenibilidad': 24,
    'Logística y Transporte': 25,
    'Mecánica Automotriz': 26,
    'Oficios Técnicos Especializados': 27,
    'Idiomas y Comunicación': 28,
    'Emprendimiento y Negocios': 29,
    'Calidad y Procesos': 30,
    # Mapeos adicionales para nombres generales que usa el modelo
    'Ciencias de la Salud': 13,  # Mapea a Enfermería y Auxiliares de Salud
    'Ingeniería': 8,  # Mapea a Sistemas e Informática
    'Ciencias Sociales': 18,  # Mapea a Psicología y Trabajo Social
    'Artes': 20,  # Mapea a Arte y Cultura
    'Ciencias Exactas': 8,  # Mapea a Sistemas e Informática
}

def get_area_id(area_name: str) -> int:
    """Obtiene el ID del área basado en el nombre, con fallback a ID 1 si no se encuentra"""
    return AREA_NAME_TO_ID.get(area_name, 1)  # Default a Administración y Gestión Empresarial

def get_latest_model(db: Session, model_type: ModelType) -> Optional[ModelVersionDB]:
    """Get the latest trained model of specified type"""
    return db.query(ModelVersionDB).filter(
        ModelVersionDB.model_type == model_type
    ).order_by(ModelVersionDB.created_at.desc()).first()

def load_model(model_path: str, model_type: ModelType):
    """Load model from cache or disk"""
    # Clear cache to ensure fresh loading
    model_cache.clear()
    
    cache_key = f"{model_type}_{model_path}"
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    try:
        if model_type == ModelType.KNN:
            model = KNNModel()
        elif model_type == ModelType.NEURAL_NETWORK:
            # For neural network, use tensorflow framework and load metadata file
            model = NeuralNetworkModel(framework='tensorflow')
            # Convert .keras path to _metadata.joblib path
            if model_path.endswith('.keras'):
                metadata_path = model_path.replace('.keras', '_metadata.joblib')
                logger.info(f"Converting path: {model_path} -> {metadata_path}")
            else:
                metadata_path = model_path
                logger.info(f"Using path as-is: {metadata_path}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if model_type == ModelType.NEURAL_NETWORK:
            logger.info(f"Loading neural network model from: {metadata_path}")
            model.load_model(metadata_path)
        else:
            model.load_model(model_path)
        model_cache[cache_key] = model
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

def prepare_prediction_input(input_data: PredictionInput) -> np.ndarray:
    """Prepare input data for prediction"""
    # Convert input to DataFrame for preprocessing - matching dataset columns exactly
    data_dict = {
        'matematicas': input_data.matematicas,
        'lectura_critica': input_data.lectura_critica,
        'ciencias_naturales': input_data.ciencias_naturales,
        'sociales_ciudadanas': input_data.sociales_ciudadanas,
        'ingles': input_data.ingles,
        'dimension_1_logico_matematico': input_data.dimension_1_logico_matematico,
        'dimension_2_comprension_comunicacion': input_data.dimension_2_comprension_comunicacion,
        'dimension_3_pensamiento_cientifico': input_data.dimension_3_pensamiento_cientifico,
        'dimension_4_analisis_social_humanistico': input_data.dimension_4_analisis_social_humanistico,
        'dimension_5_creatividad_innovacion': input_data.dimension_5_creatividad_innovacion,
        'dimension_6_liderazgo_trabajo_equipo': input_data.dimension_6_liderazgo_trabajo_equipo,
        'dimension_7_pensamiento_critico': input_data.dimension_7_pensamiento_critico,
        'dimension_8_adaptabilidad_aprendizaje': input_data.dimension_8_adaptabilidad_aprendizaje
    }
    
    # Create DataFrame with single row
    df = pd.DataFrame([data_dict])
    
    # Convert to numpy array (basic preprocessing)
    return df.values

def save_prediction_to_db(db: Session, input_data: PredictionInput, predictions: List[AreaPrediction], 
                         model_version: str, model_type: str, user_email: Optional[str] = None) -> Prediction:
    """Save prediction results to database"""
    prediction_record = Prediction(
        user_email=user_email,
        input_data=input_data.dict(),
        predictions=[pred.dict() for pred in predictions],
        model_type=model_type,
        model_version=model_version,
        created_at=datetime.now(timezone.utc)
    )
    
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)
    
    return prediction_record

@router.post("/predictions/predict", response_model=PredictionResponse)
async def make_prediction(
    input_data: PredictionInput,
    db: Session = Depends(get_db)
):
    """Make prediction for user input"""
    # Get model_type and user_email from input_data
    model_type = input_data.model_type
    user_email = input_data.user_email
    
    # Validate model_type
    if model_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type is required"
        )
    
    try:
        # Get latest model of specified type
        model_version_record = get_latest_model(db, model_type)
        if not model_version_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained {model_type} model found"
            )
        
        # Load model
        model = load_model(str(model_version_record.file_path), model_type)
        
        # Prepare input data
        X_input = prepare_prediction_input(input_data)
        
        # Make prediction
        if model_type == ModelType.KNN:
            prediction = model.predict(X_input)
            probabilities = model.predict_proba(X_input) if hasattr(model.model, 'predict_proba') else None
        else:  # Neural Network
            prediction = model.predict(X_input)
            probabilities = model.predict_proba(X_input) if hasattr(model, 'predict_proba') else None
        
        # Create area predictions
        area_predictions = []
        if probabilities is not None and len(probabilities) > 0:
            # Get class names (assuming they're stored or can be derived)
            class_names = ['Ciencias de la Salud', 'Ingeniería', 'Ciencias Sociales', 'Artes', 'Ciencias Exactas']
            probs = probabilities[0] if len(probabilities.shape) > 1 else probabilities
            
            for i, (area, prob) in enumerate(zip(class_names, probs)):
                area_predictions.append(AreaPrediction(
                    area=area,
                    area_id=get_area_id(area),
                    percentage=float(prob * 100),  # Convert to percentage
                    confidence=float(prob)
                ))
        else:
            # Fallback for models without probability prediction
            predicted_area = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
            area_predictions.append(AreaPrediction(
                area=str(predicted_area),
                area_id=get_area_id(str(predicted_area)),
                percentage=100.0,
                confidence=1.0
            ))
        
        # Sort by probability (descending)
        area_predictions.sort(key=lambda x: x.percentage, reverse=True)
        
        # Save prediction to database
        prediction_record = save_prediction_to_db(
            db, input_data, area_predictions, str(model_version_record.version), str(model_type), user_email
        )
        
        # Refresh the record to get the actual values
        db.refresh(prediction_record)
        
        # Get the actual values from the refreshed record
        record_id = int(prediction_record.id)
        record_created_at = prediction_record.created_at
        
        return PredictionResponse(
            id=record_id,
            user_email=user_email,
            predictions=area_predictions,
            model_type=str(model_type),
            model_version=str(model_version_record.version),
            processing_time=0.0,  # TODO: Calculate actual processing time
            confidence_score=area_predictions[0].confidence if area_predictions else 0.0,
            created_at=record_created_at
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/predictions/history/{user_email}")
async def get_prediction_history(
    user_email: str, 
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get prediction history for a user"""
    try:
        # Query predictions for the user
        predictions = db.query(Prediction).filter(
            Prediction.user_email == user_email
        ).order_by(
            Prediction.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        # Get total count
        total_count = db.query(Prediction).filter(
            Prediction.user_email == user_email
        ).count()
        
        # Format response
        prediction_history = []
        for pred in predictions:
            prediction_history.append({
                "prediction_id": pred.id,
                "input_data": pred.input_data,
                "predictions": pred.predictions,
                "model_version": pred.model_version,
                "timestamp": pred.created_at,
                "top_recommendation": pred.predictions[0] if pred.predictions is not None and isinstance(pred.predictions, (list, tuple)) and len(pred.predictions) > 0 else None
            })
        
        return {
            "user_email": user_email,
            "predictions": prediction_history,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )

@router.get("/predictions/stats")
async def get_prediction_stats(db: Session = Depends(get_db)):
    """Get prediction statistics"""
    try:
        # Total predictions
        total_predictions = db.query(Prediction).count()
        
        # Unique users
        unique_users = db.query(Prediction.user_email).filter(
            Prediction.user_email.isnot(None)
        ).distinct().count()
        
        # Predictions by model type (from model_version)
        model_stats = {}
        predictions_with_models = db.query(Prediction.model_version).all()
        for (model_version,) in predictions_with_models:
            if model_version:
                model_type = model_version.split('_')[0] if '_' in model_version else 'unknown'
                model_stats[model_type] = model_stats.get(model_type, 0) + 1
        
        # Recent predictions (last 7 days)
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent_predictions = db.query(Prediction).filter(
            Prediction.created_at >= seven_days_ago
        ).count()
        
        # Most common predicted areas
        area_stats = {}
        all_predictions = db.query(Prediction.predictions).all()
        for (predictions_json,) in all_predictions:
            if predictions_json:
                for pred in predictions_json:
                    if isinstance(pred, dict) and 'area_conocimiento' in pred:
                        area = pred['area_conocimiento']
                        area_stats[area] = area_stats.get(area, 0) + 1
        
        # Sort areas by frequency
        top_areas = sorted(area_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_predictions": total_predictions,
            "unique_users": unique_users,
            "recent_predictions_7d": recent_predictions,
            "predictions_by_model": model_stats,
            "top_predicted_areas": [{
                "area": area,
                "count": count,
                "percentage": round((count / total_predictions) * 100, 2) if total_predictions > 0 else 0
            } for area, count in top_areas],
            "average_predictions_per_user": round(total_predictions / unique_users, 2) if unique_users > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error retrieving prediction stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction statistics: {str(e)}"
        )

@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Delete a specific prediction"""
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        db.delete(prediction)
        db.commit()
        
        return {
            "message": "Prediction deleted successfully",
            "prediction_id": prediction_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prediction: {str(e)}"
        )
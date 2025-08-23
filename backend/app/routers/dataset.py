from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import json

from app.database.connection import get_db
from app.database.models import Student, KnowledgeArea
from app.schemas.schemas import DatasetStatus, DatasetGenerationRequest
from app.core.config import settings

# Add ml_models to path
sys.path.append(str(Path(__file__).parent.parent.parent / "ml_models"))
from ml_models.dataset_generator import DimensionalDatasetGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

# Global variable to track generation status
generation_status = {
    "is_generating": False,
    "progress": 0,
    "message": "Ready",
    "last_generated": None,
    "error": None
}

@router.get("/dataset/status", response_model=DatasetStatus)
async def get_dataset_status(db: Session = Depends(get_db)):
    """Get current dataset status and statistics"""
    try:
        # Count total students in database
        total_records = db.query(func.count(Student.id)).scalar() or 0
        
        # Get last updated timestamp
        last_student = db.query(Student).order_by(Student.created_at.desc()).first()
        last_updated = last_student.created_at if last_student else None
        
        # Check if CSV file exists
        csv_path = settings.DATA_DIR / "datasets" / "dataset_estudiantes.csv"
        csv_exists = csv_path.exists()
        csv_records = 0
        
        if csv_exists:
            try:
                df = pd.read_csv(csv_path)
                csv_records = len(df)
            except Exception as e:
                logger.warning(f"Error reading CSV: {e}")
        
        # Determine overall status
        if generation_status["is_generating"]:
            dataset_status = "generating"
        elif total_records > 0 or csv_records > 0:
            dataset_status = "ready"
        else:
            dataset_status = "empty"
        
        return DatasetStatus(
            status=dataset_status,
            total_records=total_records,
            csv_records=csv_records,
            last_updated=last_updated.isoformat() if last_updated is not None else None,
            is_generating=generation_status["is_generating"],
            generation_progress=generation_status["progress"],
            generation_message=generation_status["message"],
            csv_exists=csv_exists,
            error_message=generation_status["error"]
        )
        
    except Exception as e:
        logger.error(f"Error getting dataset status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset status: {str(e)}"
        )

async def generate_dataset_background(num_students: int, save_to_db: bool, db_session: Session):
    """Background task for dataset generation"""
    try:
        generation_status["is_generating"] = True
        generation_status["progress"] = 0
        generation_status["message"] = "Initializing dataset generator..."
        generation_status["error"] = None
        
        # Initialize generator
        generator = DimensionalDatasetGenerator(data_path=str(settings.DATA_DIR))
        
        generation_status["progress"] = 10
        generation_status["message"] = "Loading reference data..."
        
        # Load data
        generator.load_data()
        
        generation_status["progress"] = 20
        generation_status["message"] = f"Generating {num_students} synthetic students..."
        
        # Generate dataset
        df = generator.generate_dataset(num_students=num_students)
        
        generation_status["progress"] = 70
        generation_status["message"] = "Saving dataset to CSV..."
        
        # Save to CSV
        generator.save_dataset(df, for_training=True)
        
        generation_status["progress"] = 80
        
        if save_to_db:
            generation_status["message"] = "Saving to database..."
            
            # Clear existing students if requested
            db_session.query(Student).delete()
            
            # Convert DataFrame to Student objects
            students_to_add = []
            for _, row in df.iterrows():
                student = Student(
                    email=f"student_{row.name}@talentai.com",
                    age=row.get('edad', 20),
                    gender=row.get('genero', 'No especificado'),
                    academic_level=row.get('nivel_academico', 'Pregrado'),
                    comunicacion=row.get('comunicacion', 0),
                    trabajo_equipo=row.get('trabajo_equipo', 0),
                    liderazgo=row.get('liderazgo', 0),
                    resolucion_problemas=row.get('resolucion_problemas', 0),
                    creatividad=row.get('creatividad', 0),
                    pensamiento_critico=row.get('pensamiento_critico', 0),
                    adaptabilidad=row.get('adaptabilidad', 0),
                    gestion_tiempo=row.get('gestion_tiempo', 0),
                    extraversion=row.get('extraversion', 0),
                    amabilidad=row.get('amabilidad', 0),
                    responsabilidad=row.get('responsabilidad', 0),
                    neuroticismo=row.get('neuroticismo', 0),
                    apertura=row.get('apertura', 0),
                    matematicas=row.get('matematicas', 0),
                    ciencias=row.get('ciencias', 0),
                    tecnologia=row.get('tecnologia', 0),
                    artes=row.get('artes', 0),
                    humanidades=row.get('humanidades', 0),
                    ciencias_sociales=row.get('ciencias_sociales', 0),
                    area_conocimiento=str(row.get('area_conocimiento', 1))
                )
                students_to_add.append(student)
            
            # Batch insert
            db_session.add_all(students_to_add)
            db_session.commit()
        
        generation_status["progress"] = 100
        generation_status["message"] = "Dataset generation completed successfully"
        generation_status["last_generated"] = datetime.now(timezone.utc).isoformat()
        
    except Exception as e:
        logger.error(f"Error in background dataset generation: {e}")
        generation_status["error"] = str(e)
        generation_status["message"] = f"Generation failed: {str(e)}"
        db_session.rollback()
    finally:
        generation_status["is_generating"] = False
        db_session.close()

@router.post("/dataset/generate")
async def generate_dataset(
    background_tasks: BackgroundTasks,
    request: DatasetGenerationRequest,
    db: Session = Depends(get_db)
):
    """Generate or update training dataset"""
    try:
        # Check if generation is already in progress
        if generation_status["is_generating"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset generation already in progress"
            )
        
        # Validate parameters
        if request.num_students < 100 or request.num_students > 50000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of students must be between 100 and 50,000"
            )
        
        # Start background generation
        background_tasks.add_task(
            generate_dataset_background,
            request.num_students,
            request.save_to_database,
            db
        )
        
        return {
            "status": "started",
            "message": f"Dataset generation started for {request.num_students} students",
            "num_students": request.num_students,
            "save_to_database": request.save_to_database,
            "estimated_time_minutes": max(1, request.num_students // 1000)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting dataset generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting dataset generation: {str(e)}"
        )

@router.post("/dataset/validate")
async def validate_dataset(db: Session = Depends(get_db)):
    """Validate dataset integrity and format"""
    try:
        validation_results = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check CSV file
        csv_path = settings.DATA_DIR / "datasets" / "dataset_estudiantes.csv"
        if not csv_path.exists():
            validation_results["errors"].append("CSV file not found")
            validation_results["status"] = "invalid"
        else:
            try:
                df = pd.read_csv(csv_path)
                
                # Basic validation
                required_columns = [
                    'matematicas', 'lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas', 'ingles',
                    'dimension_1_logico_matematico', 'dimension_2_comprension_comunicacion', 
                    'dimension_3_pensamiento_cientifico', 'dimension_4_analisis_social_humanistico',
                    'dimension_5_creatividad_innovacion', 'dimension_6_liderazgo_trabajo_equipo',
                    'dimension_7_pensamiento_critico', 'dimension_8_adaptabilidad_aprendizaje',
                    'area_conocimiento'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_results["errors"].append(f"Missing columns: {missing_columns}")
                    validation_results["status"] = "invalid"
                
                # Check data ranges
                dimension_columns = [
                    'dimension_1_logico_matematico', 'dimension_2_comprension_comunicacion', 
                    'dimension_3_pensamiento_cientifico', 'dimension_4_analisis_social_humanistico',
                    'dimension_5_creatividad_innovacion', 'dimension_6_liderazgo_trabajo_equipo',
                    'dimension_7_pensamiento_critico', 'dimension_8_adaptabilidad_aprendizaje'
                ]
                
                for col in dimension_columns:
                    if col in df.columns:
                        if df[col].min() < 1.0 or df[col].max() > 5.0:
                            validation_results["warnings"].append(
                                f"Column {col} has values outside expected range [1.0, 5.0]"
                            )
                
                # Statistics
                validation_results["statistics"] = {
                    "total_records": int(len(df)),
                    "unique_areas": int(df['area_conocimiento'].nunique()) if 'area_conocimiento' in df.columns else 0,
                    "missing_values": int(df.isnull().sum().sum()),
                    "duplicate_records": int(df.duplicated().sum())
                }
                
            except Exception as e:
                validation_results["errors"].append(f"Error reading CSV: {str(e)}")
                validation_results["status"] = "invalid"
        
        # Check database records
        try:
            db_count = db.query(func.count(Student.id)).scalar() or 0
            validation_results["statistics"]["database_records"] = int(db_count)
            
            if db_count == 0:
                validation_results["warnings"].append("No records found in database")
                
        except Exception as e:
            validation_results["errors"].append(f"Database validation error: {str(e)}")
            validation_results["status"] = "invalid"
        
        # Final status determination
        if validation_results["errors"]:
            validation_results["status"] = "invalid"
        elif validation_results["warnings"]:
            validation_results["status"] = "valid_with_warnings"
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating dataset: {str(e)}"
        )
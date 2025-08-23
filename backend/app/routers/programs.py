from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from app.database.connection import get_db
from app.database.models import Prediction, AcademicProgram, KnowledgeArea, ProgramArea

logger = logging.getLogger(__name__)
router = APIRouter()

# Response Models
def format_program_response(program: AcademicProgram, area: Optional[KnowledgeArea] = None) -> Dict[str, Any]:
    """Format a program for API response"""
    return {
        "id": program.id,
        "name": program.name,
        "institution": program.institution,
        "academic_level": program.academic_level,
        "modality": program.modality,
        "duration": program.duration,
        "city": program.city,
        "department": program.department,
        "country": program.country,
        "description": program.description,
        "requirements": program.requirements,
        "website": program.website,
        "is_active": program.is_active,
        "created_at": program.created_at.isoformat() if program.created_at is not None else None,
        "updated_at": program.updated_at.isoformat() if program.updated_at is not None else None,
        "knowledge_area": {
            "id": area.id if area else None,
            "name": area.name if area else None,
            "description": area.description if area else None,
            "code": area.code if area else None
        } if area else None
    }

def format_prediction_response(prediction: Prediction) -> Dict[str, Any]:
    """Format a prediction for API response"""
    return {
        "id": prediction.id,
        "user_email": prediction.user_email,
        "predictions": prediction.predictions,
        "model_type": prediction.model_type,
        "model_version": prediction.model_version,
        "processing_time": prediction.processing_time,
        "confidence_score": prediction.confidence_score,
        "created_at": prediction.created_at.isoformat() if prediction.created_at is not None else None,
        "input_data": prediction.input_data
    }

@router.get("/programs/search")
async def search_programs_advanced(
    area_id: Optional[int] = Query(None, description="Filter by knowledge area ID"),
    city: Optional[str] = Query(None, description="Filter by city"),
    department: Optional[str] = Query(None, description="Filter by department"),
    academic_level: Optional[str] = Query(None, description="Filter by academic level"),
    name: Optional[str] = Query(None, description="Search by program name (similarity)"),
    limit: int = Query(50, description="Number of results per page", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    db: Session = Depends(get_db)
):
    """
    Advanced program search with multiple optional filters and pagination.
    Supports filtering by area_id, location (city, department), academic_level, and name similarity.
    """
    try:
        # Base query with joins
        query = db.query(AcademicProgram).options(
            joinedload(AcademicProgram.program_areas).joinedload(ProgramArea.area)
        ).filter(AcademicProgram.is_active == True)
        
        # Apply filters
        filters = []
        
        if area_id is not None:
            # Filter by area_id through the relationship
            query = query.join(ProgramArea).filter(ProgramArea.area_id == area_id)
        
        if city:
            filters.append(AcademicProgram.city.ilike(f"%{city}%"))
        
        if department:
            filters.append(AcademicProgram.department.ilike(f"%{department}%"))
        
        if academic_level:
            filters.append(AcademicProgram.academic_level.ilike(f"%{academic_level}%"))
        
        if name:
            filters.append(AcademicProgram.name.ilike(f"%{name}%"))
        
        # Apply all filters
        if filters:
            query = query.filter(and_(*filters))
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination
        programs = query.offset(offset).limit(limit).all()
        
        # Format response
        programs_list = []
        for program in programs:
            # Get the primary knowledge area for this program
            primary_area = None
            if program.program_areas:
                # Get the area with highest relevance score or first one
                primary_program_area = max(program.program_areas, key=lambda pa: pa.relevance_score or 0)
                primary_area = primary_program_area.area
            
            programs_list.append(format_program_response(program, primary_area))
        
        return {
            "programs": programs_list,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "current_page": (offset // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit
            },
            "filters_applied": {
                "area_id": area_id,
                "city": city,
                "department": department,
                "academic_level": academic_level,
                "name": name
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced program search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search programs: {str(e)}"
        )

@router.get("/predictions/history/{user_email}")
async def get_prediction_history(
    user_email: str,
    limit: int = Query(20, description="Number of predictions per page", ge=1, le=100),
    offset: int = Query(0, description="Number of predictions to skip", ge=0),
    db: Session = Depends(get_db)
):
    """
    Get all previous predictions for a given user email with pagination.
    Returns detailed information about each prediction including input data and results.
    """
    try:
        # Validate email format (basic validation)
        if not user_email or "@" not in user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Base query for user predictions
        query = db.query(Prediction).filter(
            Prediction.user_email == user_email
        ).order_by(Prediction.created_at.desc())
        
        # Get total count
        total_count = query.count()
        
        if total_count == 0:
            return {
                "user_email": user_email,
                "predictions": [],
                "pagination": {
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "current_page": 1,
                    "total_pages": 0
                },
                "message": "No predictions found for this user"
            }
        
        # Apply pagination
        predictions = query.offset(offset).limit(limit).all()
        
        # Format predictions
        predictions_list = [format_prediction_response(pred) for pred in predictions]
        
        # Calculate statistics
        latest_prediction = predictions[0] if predictions else None
        
        return {
            "user_email": user_email,
            "predictions": predictions_list,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "current_page": (offset // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit
            },
            "statistics": {
                "total_predictions": total_count,
                "latest_prediction_date": latest_prediction.created_at.isoformat() if latest_prediction and latest_prediction.created_at is not None else None,
                "models_used": list(set(pred.model_type for pred in predictions)),
                "average_confidence": sum(pred.confidence_score for pred in predictions if pred.confidence_score is not None) / len([pred for pred in predictions if pred.confidence_score is not None]) if predictions else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction history for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction history: {str(e)}"
        )

@router.get("/programs/areas")
async def list_knowledge_areas(
    limit: int = Query(100, description="Number of areas per page", ge=1, le=200),
    offset: int = Query(0, description="Number of areas to skip", ge=0),
    db: Session = Depends(get_db)
):
    """List all available knowledge areas with pagination"""
    try:
        # Get total count
        total_count = db.query(KnowledgeArea).count()
        
        # Get areas with pagination
        areas = db.query(KnowledgeArea).offset(offset).limit(limit).all()
        
        areas_list = []
        for area in areas:
            areas_list.append({
                "id": area.id,
                "name": area.name,
                "description": area.description,
                "code": area.code
            })
        
        return {
            "areas": areas_list,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "current_page": (offset // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing knowledge areas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list knowledge areas: {str(e)}"
        )

@router.get("/programs/filters")
async def get_available_filters(db: Session = Depends(get_db)):
    """Get all available filter options for program search"""
    try:
        # Get unique cities
        cities_query = db.query(AcademicProgram.city).filter(
            AcademicProgram.city.isnot(None),
            AcademicProgram.is_active == True
        ).distinct().order_by(AcademicProgram.city)
        cities = [city[0] for city in cities_query.all() if city[0]]
        
        # Get unique departments
        departments_query = db.query(AcademicProgram.department).filter(
            AcademicProgram.department.isnot(None),
            AcademicProgram.is_active == True
        ).distinct().order_by(AcademicProgram.department)
        departments = [dept[0] for dept in departments_query.all() if dept[0]]
        
        # Get unique academic levels
        levels_query = db.query(AcademicProgram.academic_level).filter(
            AcademicProgram.academic_level.isnot(None),
            AcademicProgram.is_active == True
        ).distinct().order_by(AcademicProgram.academic_level)
        levels = [level[0] for level in levels_query.all() if level[0]]
        
        # Get knowledge areas
        areas = db.query(KnowledgeArea).order_by(KnowledgeArea.name).all()
        knowledge_areas = [{
            "id": area.id,
            "name": area.name,
            "code": area.code
        } for area in areas]
        
        # Get total programs count
        total_programs = db.query(AcademicProgram).filter(
            AcademicProgram.is_active == True
        ).count()
        
        return {
            "cities": cities[:100],  # Limit to first 100
            "departments": departments,
            "academic_levels": levels,
            "knowledge_areas": knowledge_areas,
            "total_programs": total_programs,
            "filter_counts": {
                "cities": len(cities),
                "departments": len(departments),
                "academic_levels": len(levels),
                "knowledge_areas": len(knowledge_areas)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get filter options: {str(e)}"
        )
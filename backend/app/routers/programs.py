from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from pydantic import Field
import re
import unicodedata

from app.database.connection import get_db
from app.database.models import Prediction, AcademicProgram, KnowledgeArea, ProgramArea
from app.schemas.schemas import KnowledgeArea as KnowledgeAreaSchema, BaseSchema

logger = logging.getLogger(__name__)
router = APIRouter()

# City normalization functions
def normalize_city_name(city_name: str) -> str:
    """
    Normalize city names for consistent filtering and deduplication.
    Handles cases like 'BOGOTÁ' vs 'Bogotá D.C.' -> 'BOGOTÁ'
    Returns normalized city name in UPPERCASE format WITH accents preserved.
    """
    if not city_name:
        return city_name
    
    # Convert to uppercase first to preserve accents
    normalized = city_name.upper().strip()
    
    # Remove common suffixes and variations
    suffixes_to_remove = [
        r'\s+D\.?C\.?$',  # D.C. or DC (uppercase)
        r'\s+DISTRITO\s+CAPITAL$',
        r'\s+\(.*\)$',  # Remove parentheses content
    ]
    
    for suffix_pattern in suffixes_to_remove:
        normalized = re.sub(suffix_pattern, '', normalized)
    
    # Clean up extra spaces
    normalized = ' '.join(normalized.split())
    
    return normalized

def create_city_search_conditions(city_filter: str, model_field) -> list:
    """
    Create multiple search conditions for flexible city matching.
    Returns a list of OR conditions to match various city name formats.
    All searches are case-insensitive and handle accents.
    """
    conditions = []
    
    # Original search (case insensitive)
    conditions.append(model_field.ilike(f"%{city_filter}%"))
    
    # Normalized search (uppercase with accents preserved)
    normalized_filter = normalize_city_name(city_filter)
    if normalized_filter != city_filter.upper():
        conditions.append(model_field.ilike(f"%{normalized_filter}%"))
    
    # Also search for the lowercase version
    conditions.append(model_field.ilike(f"%{city_filter.lower()}%"))
    
    # Search without accents for flexibility
    filter_without_accents = unicodedata.normalize('NFD', city_filter)
    filter_without_accents = ''.join(c for c in filter_without_accents if unicodedata.category(c) != 'Mn')
    conditions.append(model_field.ilike(f"%{filter_without_accents}%"))
    
    # Common variations for specific cities
    city_variations = {
        'bogota': ['bogotá', 'bogota d.c.', 'bogotá d.c.', 'santafe de bogota', 'BOGOTÁ', 'BOGOTA D.C.', 'BOGOTA'],
        'medellin': ['medellín', 'MEDELLÍN', 'MEDELLIN'],
        'cali': ['santiago de cali', 'SANTIAGO DE CALI'],
        'barranquilla': ['BARRANQUILLA'],
        'cartagena': ['cartagena de indias', 'CARTAGENA DE INDIAS'],
    }
    
    # Use the filter without accents for variation matching
    filter_key = filter_without_accents.lower()
    if filter_key in city_variations:
        for variation in city_variations[filter_key]:
            conditions.append(model_field.ilike(f"%{variation}%"))
    
    return conditions

def deduplicate_cities(cities_list: List[str]) -> List[str]:
    """
    Remove duplicate cities based on normalized names.
    Returns cities in consistent UPPERCASE format WITH accents preserved.
    Prioritizes versions with accents when available.
    """
    city_groups = {}
    
    # Group cities by their normalized form (without accents for comparison)
    for city in cities_list:
        # Create a key without accents for grouping
        key_without_accents = unicodedata.normalize('NFD', normalize_city_name(city))
        key_without_accents = ''.join(c for c in key_without_accents if unicodedata.category(c) != 'Mn')
        
        if key_without_accents not in city_groups:
            city_groups[key_without_accents] = []
        city_groups[key_without_accents].append(city)
    
    # Select the best representative for each group (prefer with accents)
    deduplicated = []
    for group in city_groups.values():
        # Normalize all cities in the group
        normalized_cities = [normalize_city_name(city) for city in group]
        # Prefer the version with accents (more characters usually means accents)
        best_city = max(normalized_cities, key=lambda x: (len(x), sum(1 for c in x if ord(c) > 127)))
        deduplicated.append(best_city)
    
    return sorted(deduplicated)

# Response schemas for all endpoints
class PaginationInfo(BaseSchema):
    """Pagination information schema"""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Items skipped")
    has_more: bool = Field(..., description="Whether there are more items")
    current_page: int = Field(..., description="Current page number")
    total_pages: int = Field(..., description="Total number of pages")

class KnowledgeAreaInfo(BaseSchema):
    """Knowledge area information schema"""
    id: Optional[int] = Field(None, description="Knowledge area ID")
    name: Optional[str] = Field(None, description="Knowledge area name")
    description: Optional[str] = Field(None, description="Knowledge area description")
    code: Optional[str] = Field(None, description="Knowledge area code")

class ProgramInfo(BaseSchema):
    """Academic program information schema"""
    id: int = Field(..., description="Program unique identifier")
    name: str = Field(..., description="Program name")
    institution: Optional[str] = Field(None, description="Institution offering the program")
    academic_level: Optional[str] = Field(None, description="Academic level (e.g., Pregrado, Posgrado)")
    modality: Optional[str] = Field(None, description="Study modality (e.g., Presencial, Virtual)")
    duration: Optional[str] = Field(None, description="Program duration")
    city: Optional[str] = Field(None, description="City where program is offered")
    department: Optional[str] = Field(None, description="Department/state where program is offered")
    country: Optional[str] = Field(None, description="Country where program is offered")
    description: Optional[str] = Field(None, description="Program description")
    requirements: Optional[str] = Field(None, description="Program requirements")
    website: Optional[str] = Field(None, description="Program website URL")
    is_active: bool = Field(..., description="Whether the program is currently active")
    created_at: Optional[str] = Field(None, description="Program creation timestamp (ISO format)")
    updated_at: Optional[str] = Field(None, description="Program last update timestamp (ISO format)")
    knowledge_area: Optional[KnowledgeAreaInfo] = Field(None, description="Associated knowledge area")

class FiltersApplied(BaseSchema):
    """Applied filters information schema"""
    area_id: Optional[int] = Field(None, description="Knowledge area ID filter")
    city: Optional[str] = Field(None, description="City filter")
    department: Optional[str] = Field(None, description="Department filter")
    academic_level: Optional[str] = Field(None, description="Academic level filter")
    name: Optional[str] = Field(None, description="Name search filter")

class ProgramSearchResponse(BaseSchema):
    """Response schema for program search endpoint"""
    programs: List[ProgramInfo] = Field(..., description="List of matching academic programs")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    filters_applied: FiltersApplied = Field(..., description="Filters that were applied to the search")

class PredictionInfo(BaseSchema):
    """Prediction information schema"""
    id: int = Field(..., description="Prediction unique identifier")
    user_email: str = Field(..., description="Email of the user who made the prediction")
    predictions: Dict[str, Any] = Field(..., description="Prediction results data")
    model_type: Optional[str] = Field(None, description="Type of ML model used")
    model_version: Optional[str] = Field(None, description="Version of the ML model")
    processing_time: Optional[float] = Field(None, description="Time taken to process prediction (seconds)")
    confidence_score: Optional[float] = Field(None, description="Confidence score of the prediction")
    created_at: Optional[str] = Field(None, description="Prediction creation timestamp (ISO format)")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data used for prediction")

class PredictionStatistics(BaseSchema):
    """Prediction statistics schema"""
    total_predictions: int = Field(..., description="Total number of predictions for the user")
    latest_prediction_date: Optional[str] = Field(None, description="Date of the most recent prediction (ISO format)")
    models_used: List[str] = Field(..., description="List of model types used in predictions")
    average_confidence: float = Field(..., description="Average confidence score across all predictions")

class PredictionHistoryResponse(BaseSchema):
    """Response schema for prediction history endpoint"""
    user_email: str = Field(..., description="Email of the user")
    predictions: List[PredictionInfo] = Field(..., description="List of user predictions")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    statistics: PredictionStatistics = Field(..., description="Statistical information about user predictions")
    message: Optional[str] = Field(None, description="Additional message (e.g., when no predictions found)")

class FilterCounts(BaseSchema):
    """Filter counts information schema"""
    cities: int = Field(..., description="Number of unique cities available")
    departments: int = Field(..., description="Number of unique departments available")
    academic_levels: int = Field(..., description="Number of unique academic levels available")
    knowledge_areas: int = Field(..., description="Number of knowledge areas available")

class SimpleKnowledgeArea(BaseSchema):
    """Simplified knowledge area schema for filters"""
    id: int = Field(..., description="Knowledge area ID")
    name: str = Field(..., description="Knowledge area name")
    code: Optional[str] = Field(None, description="Knowledge area code")

class FiltersResponse(BaseSchema):
    """Response schema for available filters endpoint"""
    cities: List[str] = Field(..., description="List of available cities (limited to first 100)")
    departments: List[str] = Field(..., description="List of available departments")
    academic_levels: List[str] = Field(..., description="List of available academic levels")
    knowledge_areas: List[SimpleKnowledgeArea] = Field(..., description="List of available knowledge areas")
    total_programs: int = Field(..., description="Total number of active programs")
    filter_counts: FilterCounts = Field(..., description="Count of items in each filter category")

class KnowledgeAreasResponse(BaseSchema):
    """Response schema for knowledge areas endpoint"""
    areas: List[KnowledgeAreaSchema] = Field(..., description="List of knowledge areas")
    pagination: PaginationInfo = Field(..., description="Pagination information")

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

@router.get("/programs/search", response_model=ProgramSearchResponse)
async def search_programs_advanced(
    area_id: Optional[int] = Query(None, description="Filter by knowledge area ID"),
    city: Optional[str] = Query(None, description="Filter by city"),
    department: Optional[str] = Query(None, description="Filter by department"),
    academic_level: Optional[str] = Query(None, description="Filter by academic level"),
    name: Optional[str] = Query(None, description="Search by program name (similarity)"),
    limit: int = Query(50, description="Number of results per page", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    db: Session = Depends(get_db)
) -> ProgramSearchResponse:
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
            # Use flexible city search conditions
            city_conditions = create_city_search_conditions(city, AcademicProgram.city)
            filters.append(or_(*city_conditions))
        
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
        
        # Convert programs to ProgramInfo objects
        programs_info = [ProgramInfo(**program) for program in programs_list]
        
        # Create pagination info
        pagination_info = PaginationInfo(
            total=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_count,
            current_page=(offset // limit) + 1,
            total_pages=(total_count + limit - 1) // limit
        )
        
        # Create filters applied info
        filters_applied = FiltersApplied(
            area_id=area_id,
            city=city,
            department=department,
            academic_level=academic_level,
            name=name
        )
        
        return ProgramSearchResponse(
            programs=programs_info,
            pagination=pagination_info,
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"Error in advanced program search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search programs: {str(e)}"
        )

@router.get("/predictions/history/{user_email}", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    user_email: str,
    limit: int = Query(20, description="Number of predictions per page", ge=1, le=100),
    offset: int = Query(0, description="Number of predictions to skip", ge=0),
    db: Session = Depends(get_db)
) -> PredictionHistoryResponse:
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
            pagination_info = PaginationInfo(
                total=0,
                limit=limit,
                offset=offset,
                has_more=False,
                current_page=1,
                total_pages=0
            )
            
            statistics = PredictionStatistics(
                total_predictions=0,
                latest_prediction_date=None,
                models_used=[],
                average_confidence=0.0
            )
            
            return PredictionHistoryResponse(
                user_email=user_email,
                predictions=[],
                pagination=pagination_info,
                statistics=statistics,
                message="No predictions found for this user"
            )
        
        # Apply pagination
        predictions = query.offset(offset).limit(limit).all()
        
        # Format predictions
        predictions_list = [format_prediction_response(pred) for pred in predictions]
        
        # Calculate statistics
        latest_prediction = predictions[0] if predictions else None
        
        # Convert predictions to PredictionInfo objects
        predictions_info = [PredictionInfo(**pred) for pred in predictions_list]
        
        # Create pagination info
        pagination_info = PaginationInfo(
            total=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_count,
            current_page=(offset // limit) + 1,
            total_pages=(total_count + limit - 1) // limit
        )
        
        # Create statistics using formatted prediction data
        models_used = list(set(pred["model_type"] for pred in predictions_list if pred["model_type"]))
        predictions_with_confidence = [pred for pred in predictions_list if pred["confidence_score"] is not None]
        avg_confidence = sum(pred["confidence_score"] for pred in predictions_with_confidence) / len(predictions_with_confidence) if predictions_with_confidence else 0.0
        
        statistics = PredictionStatistics(
            total_predictions=total_count,
            latest_prediction_date=latest_prediction.created_at.isoformat() if latest_prediction and latest_prediction.created_at is not None else None,
            models_used=models_used,
            average_confidence=avg_confidence
        )
        
        return PredictionHistoryResponse(
            user_email=user_email,
            predictions=predictions_info,
            pagination=pagination_info,
            statistics=statistics,
            message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction history for {user_email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction history: {str(e)}"
        )

@router.get("/programs/areas", response_model=KnowledgeAreasResponse)
async def list_knowledge_areas(
    area_id: Optional[int] = Query(None, description="Filter by specific area ID"),
    limit: int = Query(100, description="Number of areas per page", ge=1, le=200),
    offset: int = Query(0, description="Number of areas to skip", ge=0),
    db: Session = Depends(get_db)
) -> KnowledgeAreasResponse:
    """
    List all available knowledge areas with pagination and optional filtering by area_id.
    
    Returns:
    - **areas**: List of knowledge areas with id, name, description, code, and created_at
    - **pagination**: Pagination information including total count, current page, and navigation info
    
    Query Parameters:
    - **area_id**: Optional filter to get a specific area by ID
    - **limit**: Number of areas per page (1-200, default: 100)
    - **offset**: Number of areas to skip for pagination (default: 0)
    """
    try:
        # Build base query
        query = db.query(KnowledgeArea)
        
        # Apply area_id filter if provided
        if area_id is not None:
            query = query.filter(KnowledgeArea.id == area_id)
        
        # Get total count with filters applied
        total_count = query.count()
        
        # Get areas with pagination
        areas = query.offset(offset).limit(limit).all()
        
        # Convert to schema objects
        areas_list = [KnowledgeAreaSchema.from_orm(area) for area in areas]
        
        # Create pagination info
        pagination_info = PaginationInfo(
            total=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_count,
            current_page=(offset // limit) + 1,
            total_pages=(total_count + limit - 1) // limit
        )
        
        return KnowledgeAreasResponse(
            areas=areas_list,
            pagination=pagination_info
        )
        
    except Exception as e:
        logger.error(f"Error listing knowledge areas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list knowledge areas: {str(e)}"
        )

@router.get("/programs/filters", response_model=FiltersResponse)
async def get_available_filters(db: Session = Depends(get_db)) -> FiltersResponse:
    """Get all available filter options for program search"""
    try:
        # Get unique cities with deduplication
        cities_query = db.query(AcademicProgram.city).filter(
            AcademicProgram.city.isnot(None),
            AcademicProgram.is_active == True
        ).distinct().order_by(AcademicProgram.city)
        raw_cities = [city[0] for city in cities_query.all() if city[0]]
        cities = deduplicate_cities(raw_cities)
        
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
        knowledge_areas_data = [{
            "id": area.id,
            "name": area.name,
            "code": area.code
        } for area in areas]
        
        knowledge_areas = [SimpleKnowledgeArea(**area_data) for area_data in knowledge_areas_data]
        
        # Get total programs count
        total_programs = db.query(AcademicProgram).filter(
            AcademicProgram.is_active == True
        ).count()
        
        # Create filter counts
        filter_counts = FilterCounts(
            cities=len(cities),
            departments=len(departments),
            academic_levels=len(levels),
            knowledge_areas=len(knowledge_areas)
        )
        
        return FiltersResponse(
            cities=cities[:100],  # Limit to first 100
            departments=departments,
            academic_levels=levels,
            knowledge_areas=knowledge_areas,
            total_programs=total_programs,
            filter_counts=filter_counts
        )
        
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get filter options: {str(e)}"
        )
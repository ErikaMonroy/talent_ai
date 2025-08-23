from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# Enums for validation
class ModelType(str, Enum):
    KNN = "knn"
    NEURAL_NETWORK = "neural_network"

class AcademicLevel(str, Enum):
    TECNICO = "Técnico"
    PREGRADO = "Pregrado"
    POSGRADO = "Posgrado"
    ESPECIALIZACION = "Especialización"
    MAESTRIA = "Maestría"
    DOCTORADO = "Doctorado"

class Modality(str, Enum):
    PRESENCIAL = "Presencial"
    VIRTUAL = "Virtual"
    MIXTA = "Mixta"

# Base schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        use_enum_values = True

# Student schemas
class StudentBase(BaseSchema):
    email: EmailStr
    age: Optional[int] = Field(None, ge=16, le=100)
    gender: Optional[str] = Field(None, max_length=50)
    academic_level: Optional[str] = Field(None, max_length=100)
    
    # Competencias (1-5 scale)
    comunicacion: Optional[float] = Field(None, ge=1.0, le=5.0)
    trabajo_equipo: Optional[float] = Field(None, ge=1.0, le=5.0)
    liderazgo: Optional[float] = Field(None, ge=1.0, le=5.0)
    resolucion_problemas: Optional[float] = Field(None, ge=1.0, le=5.0)
    creatividad: Optional[float] = Field(None, ge=1.0, le=5.0)
    pensamiento_critico: Optional[float] = Field(None, ge=1.0, le=5.0)
    adaptabilidad: Optional[float] = Field(None, ge=1.0, le=5.0)
    gestion_tiempo: Optional[float] = Field(None, ge=1.0, le=5.0)
    
    # Dimensiones de personalidad
    extraversion: Optional[float] = Field(None, ge=1.0, le=5.0)
    amabilidad: Optional[float] = Field(None, ge=1.0, le=5.0)
    responsabilidad: Optional[float] = Field(None, ge=1.0, le=5.0)
    neuroticismo: Optional[float] = Field(None, ge=1.0, le=5.0)
    apertura: Optional[float] = Field(None, ge=1.0, le=5.0)
    
    # Intereses académicos
    matematicas: Optional[float] = Field(None, ge=1.0, le=5.0)
    ciencias: Optional[float] = Field(None, ge=1.0, le=5.0)
    tecnologia: Optional[float] = Field(None, ge=1.0, le=5.0)
    artes: Optional[float] = Field(None, ge=1.0, le=5.0)
    humanidades: Optional[float] = Field(None, ge=1.0, le=5.0)
    ciencias_sociales: Optional[float] = Field(None, ge=1.0, le=5.0)

class StudentCreate(StudentBase):
    area_conocimiento: Optional[str] = Field(None, max_length=200)

class StudentUpdate(BaseSchema):
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=16, le=100)
    area_conocimiento: Optional[str] = Field(None, max_length=200)

class Student(StudentBase):
    id: int
    area_conocimiento: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

# Prediction schemas
class PredictionInput(BaseSchema):
    user_email: EmailStr
    
    # Puntajes ICFES (0-500)
    matematicas: float = Field(..., ge=0.0, le=500.0, description="Puntaje ICFES Matemáticas (0-500)")
    lectura_critica: float = Field(..., ge=0.0, le=400.0, description="Puntaje ICFES Lectura Crítica (0-500)")
    ciencias_naturales: float = Field(..., ge=0.0, le=300.0, description="Puntaje ICFES Ciencias Naturales (0-500)")
    sociales_ciudadanas: float = Field(..., ge=0.0, le=400.0, description="Puntaje ICFES Sociales y Ciudadanas (0-500)")
    ingles: float = Field(..., ge=0.0, le=500.0, description="Puntaje ICFES Inglés (0-500)")
    
    # Dimensiones de personalidad y competencias (1-5)
    dimension_1_logico_matematico: float = Field(..., ge=1.0, le=5.0, description="Dimensión 1: Lógico Matemático (1-5)")
    dimension_2_comprension_comunicacion: float = Field(..., ge=1.0, le=5.0, description="Dimensión 2: Comprensión y Comunicación (1-5)")
    dimension_3_pensamiento_cientifico: float = Field(..., ge=1.0, le=5.0, description="Dimensión 3: Pensamiento Científico (1-5)")
    dimension_4_analisis_social_humanistico: float = Field(..., ge=1.0, le=5.0, description="Dimensión 4: Análisis Social y Humanístico (1-5)")
    dimension_5_creatividad_innovacion: float = Field(..., ge=1.0, le=5.0, description="Dimensión 5: Creatividad e Innovación (1-5)")
    dimension_6_liderazgo_trabajo_equipo: float = Field(..., ge=1.0, le=5.0, description="Dimensión 6: Liderazgo y Trabajo en Equipo (1-5)")
    dimension_7_pensamiento_critico: float = Field(..., ge=1.0, le=5.0, description="Dimensión 7: Pensamiento Crítico (1-5)")
    dimension_8_adaptabilidad_aprendizaje: float = Field(..., ge=1.0, le=5.0, description="Dimensión 8: Adaptabilidad y Aprendizaje (1-5)")
    
    # Optional model preference
    model_type: Optional[ModelType] = ModelType.KNN

class AreaPrediction(BaseSchema):
    area: str
    area_id: int = Field(..., description="ID del área de conocimiento para filtrado")
    percentage: float = Field(..., ge=0.0, le=100.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class PredictionResponse(BaseSchema):
    id: int
    user_email: EmailStr
    predictions: List[AreaPrediction]
    model_type: str
    model_version: str
    processing_time: float
    confidence_score: Optional[float]
    created_at: datetime

class Prediction(PredictionResponse):
    input_data: Dict[str, Any]

# Model Version schemas
class ModelVersionBase(BaseSchema):
    model_type: ModelType
    version: str = Field(..., max_length=50)
    training_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    training_samples: Optional[int] = Field(None, ge=0)
    hyperparameters: Optional[Dict[str, Any]] = None

class ModelVersionCreate(ModelVersionBase):
    file_path: str = Field(..., max_length=500)

class ModelVersion(ModelVersionBase):
    id: int
    file_path: str
    is_active: bool
    created_at: datetime

# Knowledge Area schemas
class KnowledgeAreaBase(BaseSchema):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    code: Optional[str] = Field(None, max_length=50)

class KnowledgeAreaCreate(KnowledgeAreaBase):
    pass

class KnowledgeArea(KnowledgeAreaBase):
    id: int
    created_at: datetime

# Academic Program schemas
class AcademicProgramBase(BaseSchema):
    name: str = Field(..., max_length=300)
    institution: str = Field(..., max_length=300)
    academic_level: Optional[AcademicLevel] = None
    modality: Optional[Modality] = None
    duration: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=200)
    country: str = Field(default="Colombia", max_length=100)
    description: Optional[str] = None
    requirements: Optional[str] = None
    website: Optional[str] = Field(None, max_length=500)

class AcademicProgramCreate(AcademicProgramBase):
    pass

class AcademicProgramUpdate(BaseSchema):
    name: Optional[str] = Field(None, max_length=300)
    institution: Optional[str] = Field(None, max_length=300)
    academic_level: Optional[AcademicLevel] = None
    modality: Optional[Modality] = None
    is_active: Optional[bool] = None

class AcademicProgram(AcademicProgramBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

# Program search and filtering schemas
class ProgramFilters(BaseSchema):
    area: Optional[str] = Field(None, description="Knowledge area filter")
    city: Optional[str] = Field(None, description="City filter")
    department: Optional[str] = Field(None, description="Department filter")
    academic_level: Optional[AcademicLevel] = Field(None, description="Academic level filter")
    modality: Optional[Modality] = Field(None, description="Modality filter")
    institution: Optional[str] = Field(None, description="Institution filter")
    limit: int = Field(default=50, ge=1, le=100, description="Number of results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")

class ProgramSearchResponse(BaseSchema):
    programs: List[AcademicProgram]
    total: int
    limit: int
    offset: int
    filters_applied: Dict[str, Any]

# Dataset schemas
class DatasetStatus(BaseSchema):
    """Dataset status response"""
    status: str  # "empty", "ready", "generating", "error"
    total_records: int
    csv_records: int = 0
    last_updated: Optional[datetime] = None
    is_generating: bool = False
    generation_progress: int = 0  # 0-100
    generation_message: str = "Ready"
    csv_exists: bool = False
    error_message: Optional[str] = None

class DatasetGenerationRequest(BaseSchema):
    """Dataset generation request"""
    num_students: int = Field(default=1000, ge=100, le=50000, description="Number of synthetic students to generate")
    save_to_database: bool = Field(default=True, description="Whether to save generated data to database")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing data")

class DatasetGenerationResponse(BaseSchema):
    status: str
    message: str
    records_processed: Optional[int] = None
    processing_time: Optional[float] = None
    validation_errors: Optional[List[str]] = None

# Training schemas
class TrainingRequestKNN(BaseSchema):
    model_type: Literal[ModelType.KNN] = ModelType.KNN
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    force_retrain: bool = Field(default=False, description="Force retraining even if recent model exists")

class TrainingRequestNN(BaseSchema):
    model_type: Literal[ModelType.NEURAL_NETWORK] = ModelType.NEURAL_NETWORK
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    force_retrain: bool = Field(default=False, description="Force retraining even if recent model exists")

class TrainingResponse(BaseSchema):
    status: str
    message: str
    model_version: Optional[str] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time: Optional[float] = None
    model_path: Optional[str] = None

# Health check schemas
class HealthCheck(BaseSchema):
    status: str
    timestamp: datetime
    version: str
    service: str

class DetailedHealthCheck(HealthCheck):
    checks: Dict[str, str]

# Response schemas
class APIResponse(BaseSchema):
    status: str
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
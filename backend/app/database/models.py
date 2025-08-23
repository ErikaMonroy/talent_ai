from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.connection import Base

class Student(Base):
    """Modelo para estudiantes del dataset"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(50))
    academic_level = Column(String(100))
    
    # Competencias (1-5 scale)
    comunicacion = Column(Float)
    trabajo_equipo = Column(Float)
    liderazgo = Column(Float)
    resolucion_problemas = Column(Float)
    creatividad = Column(Float)
    pensamiento_critico = Column(Float)
    adaptabilidad = Column(Float)
    gestion_tiempo = Column(Float)
    
    # Dimensiones de personalidad
    extraversion = Column(Float)
    amabilidad = Column(Float)
    responsabilidad = Column(Float)
    neuroticismo = Column(Float)
    apertura = Column(Float)
    
    # Intereses académicos
    matematicas = Column(Float)
    ciencias = Column(Float)
    tecnologia = Column(Float)
    artes = Column(Float)
    humanidades = Column(Float)
    ciencias_sociales = Column(Float)
    
    # Área de conocimiento predicha
    area_conocimiento = Column(String(200))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Prediction(Base):
    """Modelo para almacenar predicciones realizadas"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    
    # Input parameters
    input_data = Column(JSON, nullable=False)
    
    # Predictions (top 5 areas with percentages)
    predictions = Column(JSON, nullable=False)  # [{"area": "Ingeniería", "percentage": 85.2}, ...]
    
    # Model information
    model_type = Column(String(50), nullable=False)  # "knn" or "neural_network"
    model_version = Column(String(50), nullable=False)
    
    # Metadata
    processing_time = Column(Float)  # seconds
    confidence_score = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelVersion(Base):
    """Modelo para tracking de versiones de modelos"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False)  # "knn" or "neural_network"
    version = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Training metadata
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    training_samples = Column(Integer)
    
    # Model parameters
    hyperparameters = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class KnowledgeArea(Base):
    """Modelo para áreas de conocimiento"""
    __tablename__ = "knowledge_areas"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text)
    code = Column(String(50), unique=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AcademicProgram(Base):
    """Modelo para programas académicos"""
    __tablename__ = "academic_programs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(300), nullable=False)
    institution = Column(String(300), nullable=False)
    
    # Program details
    academic_level = Column(String(100))  # "Pregrado", "Posgrado", "Técnico", etc.
    modality = Column(String(100))  # "Presencial", "Virtual", "Mixta"
    duration = Column(String(100))
    
    # Location
    city = Column(String(200))
    department = Column(String(200))
    country = Column(String(100), default="Colombia")
    
    # Additional info
    description = Column(Text)
    requirements = Column(Text)
    website = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ProgramArea(Base):
    """Modelo para relación entre programas y áreas de conocimiento"""
    __tablename__ = "program_areas"
    
    id = Column(Integer, primary_key=True, index=True)
    program_id = Column(Integer, ForeignKey("academic_programs.id"), nullable=False)
    area_id = Column(Integer, ForeignKey("knowledge_areas.id"), nullable=False)
    
    # Relationship strength (0.0 to 1.0)
    relevance_score = Column(Float, default=1.0)
    
    # Relationships
    program = relationship("AcademicProgram", backref="program_areas")
    area = relationship("KnowledgeArea", backref="area_programs")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Competence(Base):
    """Modelo para competencias"""
    __tablename__ = "competences"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text)
    category = Column(String(100))  # "Técnica", "Blanda", "Cognitiva", etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Dimension(Base):
    """Modelo para dimensiones de personalidad"""
    __tablename__ = "dimensions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text)
    category = Column(String(100))  # "Big Five", "DISC", etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
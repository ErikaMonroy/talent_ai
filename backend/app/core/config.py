from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TalentAI Backend API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API robusta para predicción de áreas de conocimiento y recomendación de programas académicos"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Server
    PORT: int = 8000
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "talentai"
    POSTGRES_PASSWORD: str = "talentai123"
    POSTGRES_DB: str = "talentai_db"
    POSTGRES_PORT: int = 5432
    
    @property
    def DATABASE_URL(self) -> str:
        # Use SQLite for development/testing
        if self.ENVIRONMENT == "development":
            return f"sqlite:///{self.BASE_DIR}/talentai_dev.db"
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Model settings
    MODEL_VERSION_FILE: str = "model_version.json"
    MAX_MODEL_VERSIONS: int = 5
    
    # Prediction settings
    TOP_PREDICTIONS: int = 5
    PREDICTION_THRESHOLD: float = 0.1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Environment
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)
settings.LOGS_DIR.mkdir(exist_ok=True)
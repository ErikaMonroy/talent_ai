from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging

from app.database.connection import get_db, test_connection
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME
    }

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check including database connectivity"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME,
        "checks": {
            "database": "unknown",
            "models_directory": "unknown",
            "data_directory": "unknown"
        }
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
        logger.info("Database health check passed")
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
        logger.error(f"Database health check failed: {e}")
    
    # Check models directory
    try:
        if settings.MODELS_DIR.exists():
            health_status["checks"]["models_directory"] = "healthy"
        else:
            health_status["checks"]["models_directory"] = "directory not found"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["models_directory"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check data directory
    try:
        if settings.DATA_DIR.exists():
            health_status["checks"]["data_directory"] = "healthy"
        else:
            health_status["checks"]["data_directory"] = "directory not found"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["data_directory"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Return appropriate status code
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status

@router.get("/health/readiness")
async def readiness_check(db: Session = Depends(get_db)):
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if we can query the database
        db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "error": str(e)}
        )

@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive"}
#!/usr/bin/env python3
"""
Database initialization script for TalentAI Backend

This script:
1. Creates all database tables
2. Loads initial data from CSV files
3. Sets up knowledge areas, competences, dimensions, and academic programs
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import logging

# Add the parent directory to the path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.database.connection import engine, SessionLocal, Base
from app.database.models import (
    Student, Prediction, ModelVersion, KnowledgeArea, 
    AcademicProgram, ProgramArea, Competence, Dimension
)
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False

def load_knowledge_areas():
    """Load knowledge areas from CSV"""
    db = SessionLocal()
    try:
        csv_path = settings.DATA_DIR / "areas_conocimiento.csv"
        if not csv_path.exists():
            logger.warning(f"Knowledge areas CSV not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} knowledge areas...")
        
        for _, row in df.iterrows():
            # Use id_area as unique code to avoid duplicates
            area_id = row.get('id_area', '')
            area = KnowledgeArea(
                name=row.get('nombre_area', row.get('nombre', row.get('name', ''))),
                description=row.get('descripcion_area', row.get('descripcion', row.get('description', ''))),
                code=f"AREA_{area_id}" if area_id else row.get('categoria_general', row.get('codigo', row.get('code', '')))
            )
            
            try:
                db.add(area)
                db.commit()
            except IntegrityError:
                db.rollback()
                # Area already exists, skip
                continue
                
        logger.info("✅ Knowledge areas loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading knowledge areas: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def load_competences():
    """Load competences from CSV"""
    db = SessionLocal()
    try:
        csv_path = settings.DATA_DIR / "competencias.csv"
        if not csv_path.exists():
            logger.warning(f"Competences CSV not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} competences...")
        
        for _, row in df.iterrows():
            competence = Competence(
                name=row.get('competencia', row.get('nombre', row.get('name', ''))),
                description=row.get('descripcion', row.get('description', '')),
                category=row.get('categoria', row.get('category', ''))
            )
            
            try:
                db.add(competence)
                db.commit()
            except IntegrityError:
                db.rollback()
                continue
                
        logger.info("✅ Competences loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading competences: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def load_dimensions():
    """Load personality dimensions from CSV"""
    db = SessionLocal()
    try:
        csv_path = settings.DATA_DIR / "dimensiones.csv"
        if not csv_path.exists():
            logger.warning(f"Dimensions CSV not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} dimensions...")
        
        for _, row in df.iterrows():
            dimension = Dimension(
                name=row.get('dimension', row.get('nombre', row.get('name', ''))),
                description=row.get('descripcion', row.get('description', '')),
                category=row.get('nombre_completo', row.get('tipo', row.get('categoria', row.get('category', 'Big Five'))))
            )
            
            try:
                db.add(dimension)
                db.commit()
            except IntegrityError:
                db.rollback()
                continue
                
        logger.info("✅ Dimensions loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading dimensions: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def load_academic_programs():
    """Load academic programs from CSV"""
    db = SessionLocal()
    try:
        csv_path = settings.DATA_DIR / "programas.csv"
        if not csv_path.exists():
            logger.warning(f"Programs CSV not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} academic programs...")
        
        loaded_count = 0
        for _, row in df.iterrows():
            # Extract city from ubicacion field
            ubicacion = row.get('ubicacion', '')
            city = ubicacion.strip() if ubicacion else ''
            
            # Map academic level from nivel_academico
            nivel = row.get('nivel_academico', '')
            if nivel == 'Técnico':
                academic_level = 'Técnico'
            elif nivel == 'Tecnológico':
                academic_level = 'Tecnológico'
            elif nivel == 'Universitario':
                academic_level = 'Pregrado'
            elif nivel == 'Especialización':
                academic_level = 'Especialización'
            elif nivel == 'Maestría':
                academic_level = 'Maestría'
            elif nivel == 'Doctorado':
                academic_level = 'Doctorado'
            else:
                academic_level = nivel
            
            # Get values for debugging
            formato_val = row.get('formato', '')
            duracion_val = row.get('duracion_info', '')
            
            # Debug log for first few records
            if loaded_count < 3:
                logger.info(f"Debug - Row {loaded_count + 1}: formato='{formato_val}', duracion_info='{duracion_val}', ubicacion='{ubicacion}'")
                logger.info(f"Debug - Creating program with: modality='{formato_val}', duration='{duracion_val}', city='{city}'")
            
            program = AcademicProgram(
                name=row.get('nombre_programa', ''),
                institution=row.get('institucion', ''),
                academic_level=academic_level,
                modality=formato_val,  # Use 'formato' column
                duration=duracion_val,  # Use 'duracion_info' column
                city=city,  # Use processed 'ubicacion'
                department='',  # Will be populated later if needed
                country='Colombia',
                description='',
                requirements='',
                website=''
            )
            
            # Debug log for first few records after creation
            if loaded_count < 3:
                logger.info(f"Debug - Program object created: modality='{program.modality}', duration='{program.duration}', city='{program.city}'")
            
            try:
                db.add(program)
                loaded_count += 1
                
                # Commit every 1000 records to avoid memory issues
                if loaded_count % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed batch: {loaded_count} programs processed")
                    
                # Debug log for first few records
                if loaded_count <= 3:
                    db.flush()  # Flush to get the ID for debugging
                    logger.info(f"Debug - After add: program.id={program.id}, modality='{program.modality}', duration='{program.duration}', city='{program.city}'")
                    
            except IntegrityError as e:
                logger.error(f"IntegrityError for program {row.get('nombre_programa', '')}: {e}")
                db.rollback()
                loaded_count -= 1  # Adjust counter
                continue
            except Exception as e:
                logger.error(f"Unexpected error for program {row.get('nombre_programa', '')}: {e}")
                db.rollback()
                loaded_count -= 1  # Adjust counter
                continue
                
        # Final commit for any remaining records
        try:
            db.commit()
            logger.info(f"Final commit completed")
        except Exception as e:
            logger.error(f"Error in final commit: {e}")
            db.rollback()
            
        logger.info("✅ Academic programs loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading academic programs: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def load_program_areas():
    """Load program-area relationships from CSV"""
    db = SessionLocal()
    try:
        csv_path = settings.DATA_DIR / "programas_areas.csv"
        if not csv_path.exists():
            logger.warning(f"Program areas CSV not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} program-area relationships...")
        
        loaded_count = 0
        skipped_invalid_area = 0
        skipped_missing_program = 0
        skipped_missing_area = 0
        
        for _, row in df.iterrows():
            # Use direct IDs from CSV
            program_id = row.get('id_programa')
            area_id = row.get('id_area')
            
            # Skip if area_id is not a valid integer or out of range
            if area_id is None:
                skipped_invalid_area += 1
                continue
                
            try:
                area_id_int = int(area_id)
                if area_id_int < 1 or area_id_int > 30:
                    skipped_invalid_area += 1
                    continue
            except (ValueError, TypeError):
                skipped_invalid_area += 1
                continue
            
            if program_id and area_id:
                # Verify that the program and area exist
                program_exists = db.query(AcademicProgram).filter(
                    AcademicProgram.id == program_id
                ).first()
                
                if not program_exists:
                    skipped_missing_program += 1
                    continue
                
                area_exists = db.query(KnowledgeArea).filter(
                    KnowledgeArea.id == area_id_int
                ).first()
                
                if not area_exists:
                    skipped_missing_area += 1
                    continue
                
                program_area = ProgramArea(
                    program_id=program_id,
                    area_id=area_id_int,
                    relevance_score=row.get('relevancia', row.get('relevance', 1.0))
                )
                
                try:
                    db.add(program_area)
                    db.commit()
                    loaded_count += 1
                except IntegrityError:
                    db.rollback()
                    continue
                    
        logger.info(f"✅ Program-area relationships loaded: {loaded_count} successful")
        logger.info(f"   Skipped - Invalid area IDs: {skipped_invalid_area}")
        logger.info(f"   Skipped - Missing programs: {skipped_missing_program}")
        logger.info(f"   Skipped - Missing areas: {skipped_missing_area}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading program-area relationships: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def test_database_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        # Simple query to test connection
        result = db.execute(text("SELECT 1"))
        db.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("🚀 Starting TalentAI database initialization...")
    
    # Test database connection
    if not test_database_connection():
        logger.error("❌ Cannot connect to database. Please check your configuration.")
        return False
    
    # Create tables
    if not create_tables():
        logger.error("❌ Failed to create database tables")
        return False
    
    # Load initial data
    success_count = 0
    total_operations = 5
    
    if load_knowledge_areas():
        success_count += 1
    
    if load_competences():
        success_count += 1
        
    if load_dimensions():
        success_count += 1
    
    if load_academic_programs():
        success_count += 1
        
    if load_program_areas():
        success_count += 1
    
    logger.info(f"\n📊 Initialization Summary:")
    logger.info(f"✅ Successful operations: {success_count}/{total_operations}")
    
    if success_count == total_operations:
        logger.info("🎉 Database initialization completed successfully!")
        return True
    else:
        logger.warning(f"⚠️  Some operations failed. Check logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
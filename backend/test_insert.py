#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from app.database.connection import SessionLocal
from app.database.models import AcademicProgram

def test_insert():
    """Test inserting a single program with all fields"""
    db = SessionLocal()
    
    try:
        # Create a test program
        program = AcademicProgram(
            name="TEST PROGRAM",
            institution="TEST INSTITUTION",
            academic_level="Pregrado",
            modality="Presencial",
            duration="8 semestres",
            city="Bogotá",
            department="Cundinamarca",
            country="Colombia",
            description="Test program",
            requirements="Test requirements",
            website="https://test.com"
        )
        
        print(f"Before insert: modality='{program.modality}', duration='{program.duration}', city='{program.city}'")
        
        db.add(program)
        db.flush()
        
        print(f"After flush: id={program.id}, modality='{program.modality}', duration='{program.duration}', city='{program.city}'")
        
        db.commit()
        
        print(f"After commit: Record saved successfully")
        
        # Query back the record
        saved_program = db.query(AcademicProgram).filter(AcademicProgram.id == program.id).first()
        if saved_program:
            print(f"Queried back: modality='{saved_program.modality}', duration='{saved_program.duration}', city='{saved_program.city}'")
        else:
            print("ERROR: Could not query back the saved record")
            
    except Exception as e:
        print(f"ERROR: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    test_insert()
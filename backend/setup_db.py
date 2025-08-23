#!/usr/bin/env python3
"""
Quick database setup script for TalentAI Backend

Usage:
    python setup_db.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run database initialization"""
    print("🚀 Setting up TalentAI database...")
    
    # Run the initialization script
    script_path = Path(__file__).parent / "scripts" / "init_database.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        if result.returncode == 0:
            print("✅ Database setup completed successfully!")
        else:
            print("❌ Database setup failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running database setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
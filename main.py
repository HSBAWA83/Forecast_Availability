"""
Fresh Produce Supply Chain Forecasting & Optimization System
Main Application Entry Point
"""

import os
import sys
from pathlib import Path

# Create project structure
def setup_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'models',
        'api',
        'ui',
        'utils',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✓ Project structure created successfully")

def main():
    """Main application entry point"""
    print("=" * 80)
    print("Fresh Produce Supply Chain Forecasting & Optimization System")
    print("=" * 80)
    
    # Setup project structure
    setup_project_structure()
    
    print("\n📦 System Components:")
    print("  1. Data Processing & Feature Engineering")
    print("  2. Advanced Forecasting Engine (XGBoost + Prophet)")
    print("  3. Optimization & Recommendation Engine")
    print("  4. REST API Backend (FastAPI)")
    print("  5. Interactive Dashboard (Streamlit)")
    
    print("\n🚀 Quick Start:")
    print("  Step 1: Place your DILO.xlsx in the 'data' folder")
    print("  Step 2: Run: python data_processor.py")
    print("  Step 3: Run: python forecasting_engine.py")
    print("  Step 4: Run API: uvicorn api.main:app --reload")
    print("  Step 5: Run Dashboard: streamlit run ui/dashboard.py")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

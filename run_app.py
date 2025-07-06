#!/usr/bin/env python3
"""
DermAI Assistant - Startup Script
Runs both the backend FastAPI server and Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'fastapi', 'uvicorn', 'tensorflow', 'pillow', 'requests', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def run_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])
    return backend_process

def run_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", 
        "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    return frontend_process

def main():
    print("🔬 DermAI Assistant - Starting Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if model file exists
    model_path = Path("backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras")
    if not model_path.exists():
        print("❌ Model file not found!")
        print(f"Expected: {model_path}")
        print("Please make sure your trained model is in the correct location.")
        sys.exit(1)
    
    print("✅ Model file found")
    
    try:
        # Start backend
        backend_process = run_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend_process = run_frontend()
        
        print("\n" + "=" * 50)
        print("🎉 DermAI Assistant is running!")
        print("📱 Frontend: http://localhost:8501")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop both servers")
        
        # Wait for processes
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Servers stopped")
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
"""
Quick start script for MRAS

Usage:
    python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 50)
    print("Starting MRAS - Minimal RAG Agent System")
    print("=" * 50)
    print("\nAPI will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

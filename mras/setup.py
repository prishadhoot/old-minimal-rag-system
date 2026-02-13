"""
Setup script for MRAS

This script helps with initial setup and verification.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_directories():
    """Verify all required directories exist"""
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "app",
        base_dir / "data",
        base_dir / "data" / "documents",
        base_dir / "tests"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✅ Directory exists: {directory.name}")
        else:
            print(f"❌ Directory missing: {directory.name}")
            all_exist = False
    
    return all_exist


def check_env_variable():
    """Check if OPENROUTER_API_KEY is set"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"✅ OPENROUTER_API_KEY is set")
        return True
    else:
        print("⚠️  OPENROUTER_API_KEY not set")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        print("   Or on Windows: $env:OPENROUTER_API_KEY='your-key-here'")
        return False


def check_dependencies():
    """Check if key dependencies are installed (by package name, not import)"""
    import importlib.metadata
    
    # PyPI package names (may differ from import module names)
    packages = [
        "fastapi",
        "uvicorn",
        "sentence-transformers",
        "faiss-cpu",
        "httpx",
        "pypdf",
    ]
    
    all_installed = True
    for pkg in packages:
        try:
            importlib.metadata.version(pkg)
            print(f"✅ {pkg} installed")
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {pkg} not installed")
            all_installed = False
    
    if not all_installed:
        print("\n💡 Install dependencies with: pip install -r requirements.txt")
    
    return all_installed


def main():
    print("=" * 60)
    print("MRAS Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directories),
        ("Environment Variables", check_env_variable),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        results.append(check_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! Ready to run MRAS.")
        print("\nTo start the server, run:")
        print("  python run.py")
        print("\nOr:")
        print("  uvicorn app.main:app --reload")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

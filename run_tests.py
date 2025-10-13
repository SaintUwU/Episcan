"""
Run tests for EpiScan
"""
import subprocess
import sys

def run_tests():
    """Run all tests"""
    print("🧪 Running EpiScan tests...")
    print("=" * 40)
    
    try:
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--cov=app",
            "--cov=data_collection",
            "--cov-report=html",
            "--cov-report=term"
        ], check=True)
        
        print("=" * 40)
        print("✅ All tests passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("=" * 40)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it: pip install pytest pytest-cov")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


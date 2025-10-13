"""
Setup script for EpiScan
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        return False

def create_env_file():
    """Create .env file from template"""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("✓ .env file created. Please update with your API keys.")
        return True
    else:
        print("✓ .env file already exists")
        return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def initialize_database():
    """Initialize the database"""
    return run_command("python -c \"from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('Database initialized')\"", "Initializing database")

def run_tests():
    """Run tests"""
    return run_command("python -m pytest tests/ -v", "Running tests")

def main():
    """Main setup function"""
    print("🚀 Setting up EpiScan...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Create .env file
    if not create_env_file():
        print("Failed to create .env file")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Initialize database
    if not initialize_database():
        print("Failed to initialize database")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print(" Tests failed, but continuing with setup")
    
    print("=" * 50)
    print(" EpiScan setup completed!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Start the application: python app.py")
    print("3. Start the scheduler: python run_scheduler.py")
    print("4. Visit http://localhost:5000 to see the dashboard")

if __name__ == "__main__":
    main()


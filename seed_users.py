#!/usr/bin/env python3
"""
Seed script to create admin and medical practitioner users for EpiScan
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.data_models import User

def create_seed_users():
    """Create admin and medical practitioner users"""
    app = create_app()
    
    with app.app_context():
        # Check if any users exist
        if User.query.first():
            print("Users already exist in the database. Skipping user creation.")
            return
        
        # Create admin user
        admin = User(
            full_name="System Administrator",
            email="admin@episcan.ke",
            role="admin"
        )
        admin.set_password("admin123")
        
        # Create medical practitioner user
        practitioner = User(
            full_name="Dr. Jane Mwangi",
            email="jane.mwangi@episcan.ke",
            role="user"
        )
        practitioner.set_password("user123")
        
        try:
            db.session.add(admin)
            db.session.add(practitioner)
            db.session.commit()
            
            print(" Users created successfully!")
            print("\n Login Credentials:")
            print("=" * 50)
            print(" ADMIN USER:")
            print(f"   Email: {admin.email}")
            print(f"   Password: admin123")
            print(f"   Role: {admin.role.upper()}")
            print(f"   Dashboard: /admin/dashboard")
            print("\n MEDICAL PRACTITIONER:")
            print(f"   Email: {practitioner.email}")
            print(f"   Password: user123")
            print(f"   Role: {practitioner.role.upper()}")
            print(f"   Dashboard: /user/dashboard")
            print("\n  IMPORTANT: Change these passwords after first login!")
            print("=" * 50)
            
        except Exception as e:
            db.session.rollback()
            print(f" Error creating users: {str(e)}")

if __name__ == "__main__":
    create_seed_users()



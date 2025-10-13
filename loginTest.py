# loginTest.py
from app import create_app, db
from app.models import User

# Create your Flask app instance
app = create_app()

# Wrap everything in an application context
with app.app_context():
    # Look for the admin user
    admin_email = "admin@episcan.ke"
    admin_password = "Admin123"  # Replace with the password you want to test

    admin = User.query.filter_by(email=admin_email).first()

    if admin:
        print(f"Admin found: {admin.full_name} ({admin.email})")
        if admin.check_password(admin_password):
            print("Password is correct ✅")
        else:
            print("Password is incorrect ❌")
    else:
        print("Admin not found ❌")

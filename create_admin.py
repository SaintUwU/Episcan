from app import create_app, db
from app.models import User  # adjust if your models are in a different file

app = create_app()

with app.app_context():
    admin = User(
        full_name="Admin User",
        email="admin@episcan.ke",
        role="admin"
    )
    admin.set_password("your_secure_password")  # set your password here
    db.session.add(admin)
    db.session.commit()
    print("Admin user created")

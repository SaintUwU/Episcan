from app import create_app, db
from app.models import User  # adjust if your models are in a different file

app = create_app()

with app.app_context():
    user = User(
        full_name="Regular User",
        email="user@episcan.ke",
        role="user"
    )
    user.set_password("user_password")  # replace with a secure password
    db.session.add(user)
    db.session.commit()
    print("Normal user created")

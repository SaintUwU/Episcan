# save as update_password.py and run: python3 update_password.py
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash
DB_URL = "postgresql://postgres:root@localhost:5432/episcan_db"
engine = create_engine(DB_URL)

new_hash = generate_password_hash("NewStrongPassword123!")

with engine.begin() as conn:
    conn.execute(
        text('UPDATE "users" SET password = :pw WHERE email = :email'),
        {"pw": new_hash, "email": "admin@episcan.ke"}
    )
print("Password updated.")

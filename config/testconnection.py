from sqlalchemy import create_engine

engine = create_engine(
    'postgresql+psycopg2://episcan_user:root@localhost:5432/episcan_db'
)

try:
    with engine.connect() as conn:
        print("Connection successful!")
except Exception as e:
    print(" Connection failed:", e)

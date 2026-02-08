import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)

def exec_sql(sql: str, params: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def query(sql: str, params: dict):
    with engine.begin() as conn:
        return conn.execute(text(sql), params).mappings().all()

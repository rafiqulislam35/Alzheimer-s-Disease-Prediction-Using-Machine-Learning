from db import get_db

db = get_db()
print("Connected to:", db.name)

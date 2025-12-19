from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from the .env file

from db import get_db

db = get_db()
print("Connected to:", db.name)

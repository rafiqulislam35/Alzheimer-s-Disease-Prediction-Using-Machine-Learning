from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "alzheimers_app")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not set in .env")

# Create PyMongo client
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Collections
users_col = db["users"]
predictions_col = db["predictions"]
chats_col = db["chats"]
patient_history_col = db["patient_history"]

def get_db():
    return db

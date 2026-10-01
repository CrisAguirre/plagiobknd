import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "plagio")

_client = MongoClient(MONGO_URI) if MONGO_URI else None

def get_db():
    if _client is None:
        return None
    return _client[MONGO_DB]

def log_job(data: dict):
    """Auditoría: quién/cuándo/qué cambió. No bloquea si no hay Mongo."""
    try:
        db = get_db()
        if db is None:
            return
        db["jobs"].insert_one(data)
    except Exception:
        pass

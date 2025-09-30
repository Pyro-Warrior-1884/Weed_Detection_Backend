from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# API Limits
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
ACCEPTED_CONTENT_TYPES = ["image/jpeg", "image/png"]

# CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

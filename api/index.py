"""Vercel entrypoint exposing the FastAPI app instance."""

import sys
from pathlib import Path

# Add project root to sys.path so app and core can be imported in serverless
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app  # noqa: F401

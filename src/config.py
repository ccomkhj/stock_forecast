import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cache.duckdb"

DEFAULT_PROVIDER = "stooq"
DEFAULT_HORIZON = 30

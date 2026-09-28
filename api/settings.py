import os

SERVICE_DIR = os.environ.get("SERVICE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_CONFIG = os.environ.get("MODEL_CONFIG", "configs/ensemble.yaml")
DEVICE = os.environ.get("DEVICE", "")

TORCH_CACHE = os.environ.get("TORCH_CACHE", os.path.join(SERVICE_DIR, "pretrained_cache"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.join(SERVICE_DIR, "data", "gallery.sqlite"))

PCA_DIM = int(os.environ.get("PCA_DIM", "256"))

DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.30"))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "10"))

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(SERVICE_DIR, "data", "dataset"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SERVICE_DIR, "data", "outputs"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(SERVICE_DIR, "data", "uploads"))
MAX_UPLOAD_TOTAL_MB = float(os.environ.get("MAX_UPLOAD_TOTAL_MB", "8192"))

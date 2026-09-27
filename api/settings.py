"""Настройки сервиса. Всё переопределяется переменными окружения (см. docker-compose.yml)."""
import os

SERVICE_DIR = os.environ.get("SERVICE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_CONFIG = os.environ.get("MODEL_CONFIG", "configs/ensemble.yaml")
DEVICE = os.environ.get("DEVICE", "")                       # "" -> cuda, если доступна

# кеш скачиваемых претрейнов (на инференсе не используется: веса берутся из weights/)
TORCH_CACHE = os.environ.get("TORCH_CACHE", os.path.join(SERVICE_DIR, "pretrained_cache"))

# DATABASE_URL вида postgresql://user:pass@host:5432/db -> PostgresStore
# пусто -> SqliteStore (не требует отдельного сервиса)
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.join(SERVICE_DIR, "data", "gallery.sqlite"))

# размерность после PCA для хранения в БД. 0 = хранить полный эмбеддинг.
# 256 выбрано по замерам (EXPERIMENTS.md 9.1): в онлайн-режиме стоит 0.3 pt mAP@10,
# но даёт 20x по памяти и скорости, и укладывается в лимит pgvector на HNSW-индекс (2000).
PCA_DIM = int(os.environ.get("PCA_DIM", "256"))

# порог режима отказа по умолчанию (EXPERIMENTS.md 7.2); запрос может его переопределить
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.30"))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "10"))

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(SERVICE_DIR, "..", "..", "Датасет"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SERVICE_DIR, "data", "outputs"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))

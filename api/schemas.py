"""Схемы запросов и ответов. Из них FastAPI строит спецификацию OpenAPI (/openapi.json)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model: dict = Field(description="Параметры загруженной модели: конфиг, устройство, размер входа, размерность эмбеддинга")
    gallery: dict = Field(description="Состояние базы галереи: бэкенд, число объектов, размерность")


class EmbedResponse(BaseModel):
    image_id: Optional[str] = Field(None, description="Идентификатор, если был передан")
    dim: int = Field(description="Размерность эмбеддинга")
    embedding: List[float] = Field(description="L2-нормированный вектор признаков")


class Candidate(BaseModel):
    image_id: str = Field(description="Идентификатор снимка в галерее")
    vehicle_id: Optional[str] = Field(None, description="Идентификатор ТС, если он известен базе")
    score: float = Field(description="Косинусное сходство с запросом, от -1 до 1")
    accepted: bool = Field(description="Прошёл ли кандидат порог отказа")


class SearchResponse(BaseModel):
    matched: bool = Field(description="Найдено ли уверенное совпадение (режим отказа: false = отказ)")
    threshold: float = Field(description="Порог, применённый к этому запросу")
    top_k: int
    gallery_size: int
    best_score: Optional[float] = Field(None, description="Сходство лучшего кандидата")
    vehicle_id: Optional[str] = Field(None, description="ТС лучшего принятого кандидата; null при отказе")
    candidates: List[Candidate] = Field(description="Кандидаты по убыванию сходства")
    refusal_reason: Optional[str] = Field(None, description="Почему отказ, если matched=false")


class GalleryItemResult(BaseModel):
    image_id: str
    status: str = Field(description="added | skipped")
    reason: Optional[str] = None


class GalleryUpsertResponse(BaseModel):
    added: int
    skipped: int
    gallery_size: int
    items: List[GalleryItemResult] = []


class GalleryBulkRequest(BaseModel):
    csv_path: str = Field(description="CSV с колонками image_id,x,y,w,h[,vehicle_id]; путь на стороне сервиса")
    images_dir: Optional[str] = Field(None, description="Каталог с изображениями; по умолчанию DATA_ROOT/images")
    replace: bool = Field(False, description="Очистить галерею перед загрузкой. По умолчанию нет: "
                                             "повторная загрузка того же CSV просто обновит записи")
    fit_projection: bool = Field(True, description="Обучить PCA-проекцию. Нужно при первой загрузке; "
                                                  "на непустой галерее запрещено — проекция задаёт "
                                                  "пространство базы, и пересчёт обесценил бы уже "
                                                  "сохранённые векторы")
    limit: Optional[int] = Field(None, description="Обработать только первые N строк (для проверки)")


class GalleryBulkResponse(BaseModel):
    indexed: int
    failed: int
    gallery_size: int
    dim: int
    seconds: float
    projection_dim: Optional[int] = None


class GalleryStatsResponse(BaseModel):
    backend: str
    location: str
    items: int
    vehicles: Optional[int] = None
    dim: Optional[int] = None
    projection: bool = Field(description="Обучена ли PCA-проекция")


class BatchRequest(BaseModel):
    query_csv: str = Field(description="CSV запросов: image_id,x,y,w,h (путь на стороне сервиса)")
    gallery_csv: str = Field(description="CSV галереи: image_id,x,y,w,h")
    images_dir: Optional[str] = Field(None, description="Каталог с изображениями; по умолчанию DATA_ROOT/images")
    output_dir: Optional[str] = Field(None, description="Куда положить артефакты; по умолчанию OUTPUT_DIR")
    top_k: int = Field(10, ge=1, le=100)
    threshold: Optional[float] = Field(None, description="Порог режима отказа; по умолчанию из конфига")
    use_reranking: bool = Field(True, description="k-reciprocal re-ranking по всему пакету запросов (+3.4 pt mAP@10)")


class BatchResponse(BaseModel):
    n_query: int
    n_gallery: int
    accepted: int = Field(description="Сколько запросов получили кандидата (остальные - отказ)")
    seconds: float
    output_dir: str
    files: dict = Field(description="Пути к submission.csv, candidates.csv, embeddings.npy")

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


class CsvInfo(BaseModel):
    name: str
    path: str
    role: str = Field(description="query | gallery | train | other — по имени файла")
    inferred_role: bool = Field(False, description="Роль выведена не из имени, а из состава каталога: "
                                                   "когда годных CSV ровно два и опознан один, "
                                                   "второй получает парную роль")
    rows: Optional[int] = None
    has_vehicle_id: bool = Field(description="Есть ли разметка ТС: без неё галерея наполняется без vehicle_id")
    usable: bool = Field(description="Читается и содержит колонки image_id,x,y,w,h")
    problem: Optional[str] = None


class DatasetInspectResponse(BaseModel):
    root: str
    images_dir: Optional[str] = Field(None, description="Найденный каталог с изображениями")
    images: int = Field(description="Сколько файлов изображений в нём лежит")
    csvs: List[CsvInfo] = Field(description="Все CSV каталога с их ролями")
    suggested: dict = Field(description="Что будет взято по умолчанию: query_csv, gallery_csv, bulk_csv")
    problems: List[str] = Field(description="Почему каталог непригоден; пустой список — всё в порядке")


class UploadResponse(BaseModel):
    name: str = Field(description="Имя рабочего каталога; его же передают как data_dir")
    data_dir: str = Field(description="Путь на стороне сервиса, готовый для /gallery/bulk и /jobs/batch")
    received: int = Field(description="Сколько файлов принято")
    skipped: List[str] = Field(description="Что отклонено и почему: не изображение и не CSV")
    dataset: DatasetInspectResponse = Field(description="Разбор каталога после загрузки")


class WorkspaceInfo(BaseModel):
    name: str
    path: str
    images: int
    csvs: int


class GalleryBulkRequest(BaseModel):
    data_dir: Optional[str] = Field(None, description="Каталог с данными: изображения и CSV определяются "
                                                      "автоматически (см. GET /data/inspect). "
                                                      "Явные csv_path и images_dir его переопределяют")
    csv_path: Optional[str] = Field(None, description="CSV с колонками image_id,x,y,w,h[,vehicle_id]; "
                                                     "путь на стороне сервиса")
    images_dir: Optional[str] = Field(None, description="Каталог с изображениями; по умолчанию из data_dir")
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
    data_dir: Optional[str] = Field(None, description="Каталог с данными: изображения и CSV запросов/галереи "
                                                      "определяются автоматически (см. GET /data/inspect). "
                                                      "Явные query_csv, gallery_csv и images_dir его "
                                                      "переопределяют")
    query_csv: Optional[str] = Field(None, description="CSV запросов: image_id,x,y,w,h (путь на стороне сервиса)")
    gallery_csv: Optional[str] = Field(None, description="CSV галереи: image_id,x,y,w,h")
    images_dir: Optional[str] = Field(None, description="Каталог с изображениями; по умолчанию из data_dir")
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

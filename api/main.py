"""Backend сервиса формирования цифрового признака ТС.

Все методы описаны спецификацией OpenAPI: она генерируется автоматически и доступна
как /openapi.json, интерактивная документация - /docs (Swagger UI) и /redoc.

Этапы обработки из ТЗ §4 отражены в методах:
  получение  -> валидация файла и bbox во всех методах, принимающих изображение
  обработка  -> POST /embed (эмбеддинг фиксированной размерности, float32)
  анализ     -> POST /search (косинусный поиск по базе галереи)
  результат  -> топ-N кандидатов либо пустой ответ (режим отказа, matched=false)
"""
import csv
import io
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from . import settings
from .engine import get_engine
from .schemas import (BatchRequest, BatchResponse, Candidate, EmbedResponse, GalleryBulkRequest,
                      GalleryBulkResponse, GalleryItemResult, GalleryStatsResponse,
                      GalleryUpsertResponse, HealthResponse, SearchResponse)
from .store import build_store

DESCRIPTION = """
Сервис сопоставляет снимки одного транспортного средства с разных камер **без использования
государственного номера** - только по визуальным признакам (форма, цвет, тип кузова, детали).

**Как пользоваться**

1. `POST /gallery/bulk` - проиндексировать галерею из CSV (`image_id,x,y,w,h[,vehicle_id]`).
   При первой загрузке обучается PCA-проекция, в которой хранится база.
2. `POST /search` - загрузить снимок-запрос с координатами bbox и получить ранжированный
   список кандидатов. Если лучшее сходство ниже порога, ответ содержит `matched: false` -
   это **режим отказа**, сервис не выдаёт ложное совпадение.
3. `POST /jobs/batch` - пакетный прогон query/gallery: формирует `submission.csv`,
   `candidates.csv` и `embeddings.npy` в формате сдачи.
4. `POST /explain` - Grad-CAM: тепловая карта областей, которые дали сходство запроса
   с найденным кандидатом. Нужна оператору, чтобы проверить ответ, а не верить числу.

**Режимы поиска.** В онлайне (`/search`, по одному запросу) используется косинусное
сходство. В пакетном режиме (`/jobs/batch`) дополнительно применяется k-reciprocal
re-ranking - он даёт +3.4 pt mAP@10, но работает только когда запросы обрабатываются
пакетом, потому что использует связи между запросами.
"""

app = FastAPI(
    title="Vehicle ReID Service",
    version="1.0.0",
    description=DESCRIPTION,
    openapi_tags=[
        {"name": "Сервис", "description": "Состояние и параметры"},
        {"name": "Признак", "description": "Формирование эмбеддинга по изображению и bbox"},
        {"name": "Галерея", "description": "База эмбеддингов: загрузка, статистика, удаление"},
        {"name": "Поиск", "description": "Поиск по галерее с режимом отказа"},
        {"name": "Пакетная обработка", "description": "Формирование артефактов сдачи"},
        {"name": "Интерпретируемость", "description": "Почему модель считает снимки похожими"},
    ],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
def value_error_handler(request, exc: ValueError):
    """Некорректные входные данные (битый файл, несовпадение размерностей) - это 422, не 500."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})

_store = None


def store():
    global _store
    if _store is None:
        dim = settings.PCA_DIM or int(getattr(get_engine().model, "out_dim", 0))
        _store = build_store(settings.DATABASE_URL, settings.SQLITE_PATH, dim)
    return _store


def parse_bbox(bbox: Optional[str], x, y, w, h) -> Optional[List[float]]:
    """bbox можно передать строкой "x,y,w,h" или четырьмя полями."""
    if bbox:
        parts = [p.strip() for p in bbox.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 4:
            raise HTTPException(422, "bbox должен содержать 4 числа: x,y,w,h")
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            raise HTTPException(422, f"bbox содержит нечисловые значения: {bbox}")
    elif None not in (x, y, w, h):
        vals = [float(x), float(y), float(w), float(h)]
    else:
        return None
    if vals[2] <= 0 or vals[3] <= 0:
        raise HTTPException(422, f"ширина и высота bbox должны быть положительными, получено w={vals[2]}, h={vals[3]}")
    return vals


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(422, "пустой файл")
    if len(data) > settings.MAX_UPLOAD_MB * 2 ** 20:
        raise HTTPException(413, f"файл больше {settings.MAX_UPLOAD_MB} МБ")
    return data


def resolve(path: str, label: str) -> str:
    p = path if os.path.isabs(path) else os.path.join(settings.SERVICE_DIR, path)
    if not os.path.exists(p):
        raise HTTPException(404, f"{label} не найден: {p}")
    return p


def read_annotation_csv(path: str) -> pd.DataFrame:
    """CSV с заголовком или без (формат test_query.csv / test_gallery.csv)."""
    cols = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip().split(",")
    df = pd.read_csv(path) if first[0] == "image_id" else pd.read_csv(path, header=None, names=cols[:len(first)])
    missing = {"image_id", "x", "y", "w", "h"} - set(df.columns)
    if missing:
        raise HTTPException(422, f"в {os.path.basename(path)} нет колонок: {sorted(missing)}")
    return df


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["Сервис"],
         summary="Состояние сервиса, параметры модели и галереи")
def health():
    return {"status": "ok", "model": get_engine().describe(), "gallery": store().stats()}


@app.post("/embed", response_model=EmbedResponse, tags=["Признак"],
          summary="Сформировать цифровой признак (эмбеддинг) одного ТС")
async def embed(
    file: UploadFile = File(..., description="Кадр в JPEG/PNG"),
    bbox: Optional[str] = Form(None, description='Рамка ТС: "x,y,w,h" в пикселях кадра. Без неё берётся весь кадр'),
    x: Optional[float] = Form(None), y: Optional[float] = Form(None),
    w: Optional[float] = Form(None), h: Optional[float] = Form(None),
    image_id: Optional[str] = Form(None, description="Необязательный идентификатор для ответа"),
):
    eng = get_engine()
    img = eng.open_image(await read_upload(file))
    feat = eng.embed_images([(img, parse_bbox(bbox, x, y, w, h))])
    return {"image_id": image_id, "dim": int(feat.shape[1]), "embedding": feat[0].tolist()}


@app.get("/gallery/stats", response_model=GalleryStatsResponse, tags=["Галерея"],
         summary="Размер и параметры базы галереи")
def gallery_stats():
    return store().stats()


@app.post("/gallery/items", response_model=GalleryUpsertResponse, tags=["Галерея"],
          summary="Добавить один снимок в галерею")
async def gallery_add(
    file: UploadFile = File(..., description="Кадр в JPEG/PNG"),
    image_id: str = Form(..., description="Уникальный идентификатор снимка"),
    vehicle_id: Optional[str] = Form(None, description="Идентификатор ТС, если известен"),
    bbox: Optional[str] = Form(None, description='Рамка ТС: "x,y,w,h"'),
    x: Optional[float] = Form(None), y: Optional[float] = Form(None),
    w: Optional[float] = Form(None), h: Optional[float] = Form(None),
):
    eng, st = get_engine(), store()
    img = eng.open_image(await read_upload(file))
    feat = eng.embed_images([(img, parse_bbox(bbox, x, y, w, h))])
    try:
        vec = st.project(feat)
    except ValueError as e:
        raise HTTPException(409, str(e))
    st.upsert([{"image_id": image_id, "vehicle_id": vehicle_id, "source": "upload"}], vec)
    return {"added": 1, "skipped": 0, "gallery_size": st.stats()["items"],
            "items": [{"image_id": image_id, "status": "added"}]}


@app.post("/gallery/bulk", response_model=GalleryBulkResponse, tags=["Галерея"],
          summary="Проиндексировать галерею из CSV с аннотациями")
def gallery_bulk(req: GalleryBulkRequest = Body(...)):
    eng, st = get_engine(), store()
    csv_path = resolve(req.csv_path, "CSV галереи")
    images_dir = resolve(req.images_dir or os.path.join(settings.DATA_ROOT, "images"), "каталог изображений")
    df = read_annotation_csv(csv_path)
    if req.limit:
        df = df.head(req.limit)
    if df.empty:
        raise HTTPException(422, "CSV галереи пуст")

    t0 = time.time()
    paths = [os.path.join(images_dir, f"{i}.jpg") for i in df.image_id]
    bboxes = df[["x", "y", "w", "h"]].to_numpy().tolist()
    feats, ok = eng.embed_paths(paths, bboxes)
    if not ok:
        raise HTTPException(422, f"ни одно изображение не прочитано из {images_dir}")

    if req.replace:
        st.clear()
    proj_dim = None
    already = st.stats()["items"]
    want_fit = req.fit_projection and settings.PCA_DIM and settings.PCA_DIM < feats.shape[1]
    if want_fit and already and st.projection() is not None:
        raise HTTPException(409, f"в галерее уже {already} объектов, сохранённых в текущей "
                                 f"PCA-проекции. Пересчёт проекции сделал бы их несравнимыми: "
                                 f"передайте fit_projection=false, чтобы дополнить базу, или "
                                 f"replace=true, чтобы построить её заново")
    if want_fit:
        if len(feats) <= settings.PCA_DIM:
            raise HTTPException(422, f"для PCA до {settings.PCA_DIM}-d нужно больше {settings.PCA_DIM} "
                                     f"объектов галереи, получено {len(feats)}")
        st.fit_projection(feats, settings.PCA_DIM)
        proj_dim = settings.PCA_DIM
    vecs = st.project(feats)
    rows = df.iloc[ok]
    st.upsert([{"image_id": str(r.image_id),
                "vehicle_id": None if "vehicle_id" not in rows.columns or pd.isna(r.vehicle_id) else str(r.vehicle_id),
                "source": os.path.basename(csv_path)} for r in rows.itertuples()], vecs)
    return {"indexed": len(ok), "failed": len(df) - len(ok), "gallery_size": st.stats()["items"],
            "dim": int(vecs.shape[1]), "seconds": round(time.time() - t0, 2), "projection_dim": proj_dim}


@app.delete("/gallery", tags=["Галерея"], summary="Очистить галерею")
def gallery_clear():
    return {"removed": store().clear(), "gallery_size": store().stats()["items"]}


@app.delete("/gallery/items", tags=["Галерея"], summary="Удалить снимки по идентификаторам")
def gallery_delete(image_ids: List[str] = Query(..., description="Один или несколько image_id")):
    return {"removed": store().delete(image_ids), "gallery_size": store().stats()["items"]}


@app.post("/search", response_model=SearchResponse, tags=["Поиск"],
          summary="Найти ТС в галерее по снимку (с режимом отказа)")
async def search(
    file: UploadFile = File(..., description="Кадр-запрос в JPEG/PNG"),
    bbox: Optional[str] = Form(None, description='Рамка ТС: "x,y,w,h" в пикселях кадра'),
    x: Optional[float] = Form(None), y: Optional[float] = Form(None),
    w: Optional[float] = Form(None), h: Optional[float] = Form(None),
    top_k: int = Form(settings.DEFAULT_TOP_K, ge=1, le=100, description="Сколько кандидатов вернуть"),
    threshold: Optional[float] = Form(None, description="Порог режима отказа; по умолчанию из конфига"),
):
    eng, st = get_engine(), store()
    stats = st.stats()
    if stats["items"] == 0:
        raise HTTPException(409, "галерея пуста: сначала загрузите её через /gallery/bulk")
    thr = settings.DEFAULT_THRESHOLD if threshold is None else float(threshold)
    img = eng.open_image(await read_upload(file))
    feat = eng.embed_images([(img, parse_bbox(bbox, x, y, w, h))])
    try:
        found = st.search(st.project(feat)[0], top_k)
    except ValueError as e:
        raise HTTPException(409, str(e))

    cands = [Candidate(image_id=c["image_id"], vehicle_id=c.get("vehicle_id"),
                       score=c["score"], accepted=c["score"] >= thr) for c in found]
    best = cands[0] if cands else None
    matched = bool(best and best.accepted)
    return {
        "matched": matched,
        "threshold": thr,
        "top_k": top_k,
        "gallery_size": stats["items"],
        "best_score": None if best is None else best.score,
        "vehicle_id": best.vehicle_id if matched else None,
        "candidates": cands,
        "refusal_reason": None if matched else
            (f"лучшее сходство {best.score:.3f} ниже порога {thr:.2f}" if best else "галерея пуста"),
    }


@app.post("/jobs/batch", response_model=BatchResponse, tags=["Пакетная обработка"],
          summary="Сформировать submission.csv, candidates.csv и embeddings.npy")
def jobs_batch(req: BatchRequest = Body(...)):
    eng = get_engine()
    q_path = resolve(req.query_csv, "CSV запросов")
    g_path = resolve(req.gallery_csv, "CSV галереи")
    images_dir = resolve(req.images_dir or os.path.join(settings.DATA_ROOT, "images"), "каталог изображений")
    out_dir = req.output_dir or settings.OUTPUT_DIR
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(settings.SERVICE_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    thr = eng.cfg["infer"]["candidate_threshold"] if req.threshold is None else float(req.threshold)

    qdf, gdf = read_annotation_csv(q_path), read_annotation_csv(g_path)
    if qdf.empty or gdf.empty:
        raise HTTPException(422, "CSV запросов или галереи пуст")
    t0 = time.time()
    mk = lambda df: ([os.path.join(images_dir, f"{i}.jpg") for i in df.image_id],
                     df[["x", "y", "w", "h"]].to_numpy().tolist())
    qf, q_ok = eng.embed_paths(*mk(qdf))
    gf, g_ok = eng.embed_paths(*mk(gdf))
    if not q_ok or not g_ok:
        raise HTTPException(422, f"не удалось прочитать изображения из {images_dir}")
    qdf, gdf = qdf.iloc[q_ok].reset_index(drop=True), gdf.iloc[g_ok].reset_index(drop=True)

    # уверенность для режима отказа всегда по косинусу: порог откалиброван на нём,
    # а шкала re-ranked score другая (см. EXPERIMENTS.md §7.2)
    cos = qf @ gf.T
    sim = eng.rerank(qf, gf) if req.use_reranking else cos
    top_k = min(req.top_k, len(gdf))
    order = np.argsort(-sim, axis=1)[:, :top_k]
    g_ids = gdf.image_id.to_numpy()

    files = {k: os.path.join(out_dir, v) for k, v in
             (("submission", "submission.csv"), ("candidates", "candidates.csv"), ("embeddings", "embeddings.npy"))}
    np.save(files["embeddings"], np.concatenate([qf, gf], axis=0).astype(np.float32))
    with open(files["submission"], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        for i, qid in enumerate(qdf.image_id):
            wr.writerow([qid] + g_ids[order[i]].tolist())
    accepted = 0
    with open(files["candidates"], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(qdf.image_id):
            j = int(order[i, 0])
            if cos[i, j] >= thr:                    # ниже порога - пустой ответ (отказ)
                wr.writerow([qid, g_ids[j], round(float(cos[i, j]), 6)])
                accepted += 1
    return {"n_query": len(qdf), "n_gallery": len(gdf), "accepted": accepted,
            "seconds": round(time.time() - t0, 2), "output_dir": out_dir, "files": files}


def _reference_image(image_id: str, images_dir: str):
    """Изображение кандидата с диска и его рамка из аннотаций -> (image, bbox).

    Для объяснения нужен исходный кадр: в базе лежит только эмбеддинг, да ещё и
    сокращённый PCA, а Grad-CAM строится в пространстве модели.
    """
    path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, f"снимок галереи не найден на диске: {path}. "
                                 f"Объяснение требует исходного изображения кандидата.")
    bbox = None
    root = os.path.dirname(os.path.normpath(images_dir))
    for name in ("gallery_split.csv", "test_gallery.csv", "query_split.csv",
                 "test_query.csv", "train.csv"):
        csv_path = os.path.join(root, name)
        if not os.path.exists(csv_path):
            continue
        df = read_annotation_csv(csv_path)
        row = df[df.image_id.astype(str) == str(image_id)]
        if len(row):
            r = row.iloc[0]
            bbox = [float(r.x), float(r.y), float(r.w), float(r.h)]
            break
    with Image.open(path) as im:
        return im.copy(), bbox


@app.post("/explain", tags=["Интерпретируемость"], summary="Grad-CAM: почему снимки признаны похожими",
          responses={200: {"content": {"image/png": {}},
                           "description": "PNG: строка на снимок — кроп и тепловая карта рядом"}})
async def explain(
    file: UploadFile = File(..., description="Кадр-запрос в JPEG/PNG"),
    bbox: Optional[str] = Form(None, description='Рамка ТС: "x,y,w,h" в пикселях кадра'),
    x: Optional[float] = Form(None), y: Optional[float] = Form(None),
    w: Optional[float] = Form(None), h: Optional[float] = Form(None),
    reference_image_id: Optional[str] = Form(None, description="С каким объектом галереи сравнивать; "
                                                              "по умолчанию — лучший кандидат"),
    mode: str = Form("pair", description="pair — карты для обоих снимков (по умолчанию); "
                                         "side_by_side — кроп запроса и его карта; overlay — только карта запроса"),
):
    """Тепловая карта областей, определивших сходство запроса и кандидата.

    Считается градиент косинусного сходства по карте активаций последнего блока backbone
    (Grad-CAM). Сходство симметрично, поэтому в режиме `pair` строятся карты для **обоих**
    снимков: карта запроса — относительно эмбеддинга кандидата, карта кандидата —
    относительно эмбеддинга запроса. Так видно, смотрела ли модель на соответственные
    части. У ансамбля карты членов усредняются; метрики внимания — в заголовках X-Focus-*.
    """
    eng, st = get_engine(), store()
    if mode not in ("pair", "side_by_side", "overlay"):
        raise HTTPException(422, "mode должен быть pair, side_by_side или overlay")
    img = eng.open_image(await read_upload(file))
    box = parse_bbox(bbox, x, y, w, h)
    images_dir = os.path.join(settings.DATA_ROOT, "images")

    ref_id = reference_image_id
    score = None
    if ref_id is None:                       # берём лучшего кандидата из базы
        if st.stats()["items"] == 0:
            raise HTTPException(409, "галерея пуста и reference_image_id не задан: "
                                     "не с чем сравнивать")
        feat = eng.embed_images([(img, box)])
        found = st.search(st.project(feat)[0], 1)
        if not found:
            raise HTTPException(409, "в галерее нет кандидатов")
        ref_id, score = found[0]["image_id"], found[0]["score"]

    ref_img, ref_box = _reference_image(ref_id, images_dir)
    if mode == "pair":
        picture, stats = eng.explain_pair(img, box, ref_img, ref_box)
        score = stats["score"]
        focus = stats["query"]
        extra = {"X-Focus-Center-Share-Reference": f"{stats['reference']['center_share']:.3f}",
                 "X-Focus-Peak-Area-Reference": f"{stats['reference']['peak_area']:.3f}"}
    else:
        reference = eng.embed_images([(ref_img, ref_box)])[0]
        picture, focus = eng.explain(img, box, reference, mode=mode)
        stats, extra = focus, {}
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    headers = {
        "X-Reference-Image-Id": str(ref_id),
        "X-Explain-Mode": mode,
        "X-Members-Used": str(stats["members_used"]),
        "X-Focus-Center-Share": f"{focus['center_share']:.3f}",
        "X-Focus-Peak-Area": f"{focus['peak_area']:.3f}",
        **extra,
    }
    if score is not None:
        headers["X-Score"] = f"{score:.4f}"
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

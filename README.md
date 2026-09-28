# Сервис формирования цифрового признака ТС

Готовое к развёртыванию решение: backend инференса с OpenAPI, СУБД для базы эмбеддингов
и веб-интерфейс. Код модели (`reid/`) и конфиги (`configs/`) лежат внутри каталога.

**Перед первым запуском скачайте веса модели** — в репозитории их нет:
[disk.yandex.ru/client/disk/LCT](https://disk.yandex.ru/client/disk/LCT).
Оба файла положите в `weights/`, подробности — в разделе [«Веса модели»](#веса-модели).

## Запуск без Docker

```bash
python run_local.py
```

Поднимает backend и интерфейс, ждёт загрузки модели и печатает адреса. `Ctrl+C` останавливает оба.
Флаги:

```bash
python run_local.py --api-only              
python run_local.py --port 8000 --ui-port 8501
python run_local.py --data-root D:/data/Датасет
python run_local.py --reload                
```

Хранилище по умолчанию — SQLite (`data/gallery.sqlite`).
Зависимости: `pip install -r requirements-api.txt -r requirements-ui.txt`
плюс torch.

## Запуск в Docker

```bash
docker compose up --build
```

| компонент | адрес | что это |
|---|---|---|
| Веб-интерфейс | http://localhost:8501 | тонкий клиент: инструкция, пакетная обработка, поиск по фото, эксперименты |
| Swagger UI | http://localhost:8000/docs | интерактивная документация API |
| ReDoc | http://localhost:8000/redoc | альтернативный просмотр спецификации |
| Спецификация OpenAPI | http://localhost:8000/openapi.json | 


### Компоненты по отдельности

```bash
python -m uvicorn api.main:app --port 8000                       
API_URL=http://localhost:8000 streamlit run ui/app.py            
```

## Архитектура

```
              браузер (тонкий клиент)
                       │ HTTP
        ┌──────────────▼──────────────┐
        │  ui   Streamlit, 8501       │   
        └──────────────┬──────────────┘
                       │ HTTP (OpenAPI)
        ┌──────────────▼──────────────┐
        │  api  FastAPI + модель, 8000│   
        └──────────────┬──────────────┘
                       │ SQL
        ┌──────────────▼──────────────┐
        │  db   PostgreSQL + pgvector │ 
        └─────────────────────────────┘
```


Названия файлов латиницей; заголовки страниц в меню задаются в `ui/app.py` через `st.Page`.

## Методы API

| метод | назначение |
|---|---|
| `GET /health` | состояние, параметры модели и галереи |
| `POST /embed` | эмбеддинг по изображению и bbox |
| `POST /data/upload` | загрузить изображения, CSV или ZIP с компьютера пользователя |
| `GET /data/inspect` | разобрать каталог: найти изображения и CSV, определить их роли |
| `GET /data/uploads` | список ранее загруженных каталогов |
| `DELETE /data/uploads/{name}` | удалить загруженный каталог |
| `POST /gallery/bulk` | проиндексировать галерею — достаточно передать `data_dir` |
| `POST /gallery/items` | добавить один снимок |
| `GET /gallery/stats` | размер и параметры базы |
| `DELETE /gallery` | очистить базу |
| `DELETE /gallery/items` | удалить снимки по идентификаторам |
| `POST /search` | поиск по снимку с режимом отказа |
| `POST /jobs/batch` | сформировать `submission.csv`, `candidates.csv`, `embeddings.npy` |
| `POST /explain` | Grad-CAM: PNG с областями, определившими сходство |

Пример: наполнить базу и найти ТС по снимку.

```bash
curl 'http://localhost:8000/data/inspect?path=/data/dataset'

curl -X POST http://localhost:8000/gallery/bulk \
  -H 'Content-Type: application/json' \
  -d '{"data_dir":"/data/dataset","replace":true}'

curl -X POST http://localhost:8000/search \
  -F "file=@frame.jpg" -F "bbox=1309,65,609,483" -F "top_k=10" -F "threshold=0.3"
```

Если данных на сервере нет, их можно загрузить с машины пользователя — отдельными файлами
или ZIP-архивом каталога:

```bash
curl -X POST http://localhost:8000/data/upload \
  -F "files=@dataset.zip" -F "name=moi_nabor" -F "reset=true"
```

Ответ содержит `data_dir` готового рабочего каталога и его разбор. Файлы раскладываются
по типу: изображения в `images/`, CSV в корень; внутренняя структура архива значения
не имеет. Записи с путями наружу каталога и файлы посторонних типов отбрасываются.
Загруженное переживает перезапуск и доступно через `GET /data/uploads`.

Достаточно указать каталог: `GET /data/inspect` показывает, какие изображения и CSV в нём
найдены и какая у каждого роль, а `data_dir` в `/gallery/bulk` и `/jobs/batch` использует
тот же разбор. Роль CSV определяется по имени (`query` / `gallery`, в том числе по-русски
и в транслитерации); если годных CSV ровно два и опознан один — второй получает парную
роль. Изображения ищутся в `images/`, иначе в самом каталоге или в том его подкаталоге,
где их больше всего. Любой выбор переопределяется явными `csv_path`, `query_csv`,
`gallery_csv`, `images_dir`.

Ответ при отказе:

```json
{ "matched": false, "threshold": 0.3, "best_score": 0.206,
  "vehicle_id": null, "refusal_reason": "лучшее сходство 0.206 ниже порога 0.30",
  "candidates": [ {"image_id": "...", "score": 0.206, "accepted": false} ] }
```

## Интерпретируемость

`POST /explain` строит тепловую карту Grad-CAM: дифференцируется **косинусное сходство
эмбеддинга запроса с эмбеддингом кандидата** по карте активаций последнего блока backbone.

Сходство симметрично, поэтому по умолчанию (`mode=pair`) карта строится для **обоих**
снимков: карта запроса — относительно эмбеддинга кандидата, карта кандидата — относительно
эмбеддинга запроса.

У ансамбля карты обоих членов усредняются, подложка обесцвечивается .

```bash
curl -X POST http://localhost:8000/explain   -F "file=@frame.jpg" -F "bbox=1309,65,609,483"   -F "reference_image_id=29a63a51405d4384ba62046f24e3d5f9" -o gradcam.png
```

В заголовках ответа: `X-Reference-Image-Id`, `X-Score`, `X-Explain-Mode`, `X-Members-Used`,
`X-Focus-Center-Share` и `X-Focus-Peak-Area` для запроса, а в режиме `pair` ещё и
`X-Focus-Center-Share-Reference`, `X-Focus-Peak-Area-Reference` для кандидата.

В интерфейсе — кнопка «Построить карту» на странице поиска, с выбором кандидата
и режима отображения. Реализация — `reid/gradcam.py`.

## СУБД

Зависит от `DATABASE_URL`:

| режим | что используется | когда |
|---|---|---|
| `DATABASE_URL` не задан | **SQLite** — файл `data/gallery.sqlite`, поиск матмулом в numpy | `run_local.py`, разработка, демонстрация без Docker |
| `DATABASE_URL=postgresql://…` | **PostgreSQL + pgvector**, HNSW-индекс по косинусу | `docker compose up` — так задано в `docker-compose.yml` |


## Настройки (переменные окружения)

| переменная | по умолчанию | смысл |
|---|---|---|
| `MODEL_CONFIG` | `configs/ensemble.yaml` | конфиг модели |
| `DATABASE_URL` | — | DSN PostgreSQL; пусто → SQLite |
| `SQLITE_PATH` | `data/gallery.sqlite` | файл базы для SQLite |
| `PCA_DIM` | `256` | размерность хранения; `0` — полный эмбеддинг |
| `DEFAULT_THRESHOLD` | `0.30` | порог режима отказа |
| `DATA_ROOT` | `data/dataset` | каталог с данными по умолчанию; можно загрузить через `POST /data/upload` |
| `OUTPUT_DIR` | `data/outputs` | куда писать артефакты пакетной обработки |
| `SERVICE_DIR` | каталог сервиса | база для относительных путей |
| `DEVICE` | автоматически | `cuda` или `cpu` |
| `UPLOAD_DIR` | `data/uploads` | куда складываются загруженные наборы |
| `MAX_UPLOAD_TOTAL_MB` | `8192` | предел суммарного размера одной загрузки |
| `API_URL` (для ui) | `http://localhost:8000` | адрес backend |

## Режимы поиска

| режим | как ищет | mAP@10 | задержка при базе 10 тыс. |
|---|---|---|---|
| онлайн (`POST /search`) | косинус по PCA-256 | 0.857 | 0.43 мс |
| пакетный (`POST /jobs/batch`) | косинус + k-reciprocal re-ranking | **0.894** | 0.26 мс/запрос |

## Порог режима отказа

По умолчанию **0.30** — максимум F1 на размеченной выборке. Порог задаётся в запросе,
поэтому под сценарий можно выбрать другую рабочую точку:

| режим эксплуатации | порог | точность | полнота | доля запросов с ответом |
|---|---|---|---|---|
| подсказка оператору | 0.30 | 0.78 | 0.90 | 76% |
| баланс | 0.35 | 0.84 | 0.83 | 67% |
| высокая точность | 0.45 | 0.91 | 0.65 | 50% |
| решение без человека | 0.60 | 1.00 | 0.39 | 28% |


## Веса модели

**Весов нет в репозитории — их нужно скачать отдельно:**

### https://disk.yandex.ru/client/disk/LCT

Оба файла кладутся в каталог `weights/` рядом с `api/` и `configs/`:

```
service/
  weights/
    bestv10_512.pth
    r101_ibn_256.pth
```

| файл | размер | что это |
|---|---|---|
| `bestv10_512.pth` | 92 МБ | R101-IBN + 3 PCB-полосы + GeM, вход 512px |
| `r101_ibn_256.pth` | 92 МБ | R101-IBN, вход 256px |

Без них сервис не запустится: при старте загружается ансамбль из
`configs/ensemble.yaml`, и оба чекпоинта обязательны.

Веса хранятся в **fp16** — вдвое меньше исходных 184 МБ при том же качестве:
при загрузке они приводятся к fp32, косинус между эмбеддингами fp16- и fp32-версий
равен 0.999998, метрики эталонного протокола совпадают до четвёртого знака
(mAP@10 0.8941, Rank-1 0.8425, Rank-5 0.9669).

## Внешние ресурсы

Веса моделей лежат в `weights/` и загружаются целиком из чекпоинтов — при работе
сервиса обращений в сеть нет, модель строится с `pretrained=False`. Ниже — источники
весов, использованные при обучении.

- **ResNet50-IBN-a / ResNet101-IBN-a** — официальные ImageNet-веса IBN-Net
  (Pan et al., ECCV 2018; github.com/XingangPan/IBN-Net, releases v1.0). Запасной
  вариант — `torchvision.models.resnet50` `IMAGENET1K_V2`.
- **Vehicle-ReID претрейн** — `veri_sbs_R50-ibn.pth` из model zoo fast-reid
  (JDAI-CV/fast-reid, Apache-2.0), обучен на публичном VeRi-776. Проверен, в итоговый
  ансамбль не вошёл (`EXPERIMENTS.md` §4.3).
- **CLIP RN50** — image-encoder OpenAI CLIP (MIT) через `timm`
  (`resnet50_clip_gap.openai`). Проверен, в итоговый ансамбль не вошёл (`EXPERIMENTS.md` §4.4).

Состав итогового ансамбля — `configs/ensemble.yaml`: `bestv10_512` и `r101_ibn_256`,
обе обучены нами. Готовые веса скачиваются по ссылке из раздела «Веса модели».

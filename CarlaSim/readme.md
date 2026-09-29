# CARLA Vehicle ReID Dataset Generator

Скрипт для генерации синтетического датасета автомобилей в симуляторе CARLA для расширения датасета реидентификации транспортных средств.

Генерируемый датасет содержит изображения автомобилей в разных ракурсах, при разных погодных условиях и освещении, а также разметку 2D bounding boxes и CSV-описания для обучения и тестирования.

---

## Особенности

- Генерация изображений в разрешении `1920x1080`.
- Реалистичные ракурсы камер:
  - вид спереди;
  - вид сзади;
  - боковые ракурсы;
  - дальние камеры;
- Разнообразие условий:
  - день;
  - облачность;
  - дождь;
  - мокрая дорога;
  - закат;
  - туман;
  - вечернее освещение.
- Автоматическое получение 2D bounding boxes через проекцию 3D-боксов автомобиля из CARLA.
- Разделение данных:
  - `train.csv`;
  - `test_query.csv`;
  - `test_gallery.csv`.

---

## Структура репозитория

```text
.
├── generate_carla_reid_dataset.py   # основной скрипт генерации датасета
├── requirements.txt                 # зависимости Python
└── README.md                        # описание проекта
```

---

## Требования

- Запущенный сервер CARLA.
- CARLA Python API, соответствующий версии сервера.
- Python 3.8+

---

## Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## Запуск

Сначала запустите сервер CARLA.
Для получения фотореалистичных изображений используйте NVIDIA Cosmos Transfer1 либо NVIDIA NuRec. подробнее https://carla.org/2025/09/16/release-0.9.16/ 

Затем выполните скрипт генерации:

```bash
python generate_carla_reid_dataset.py
```

По умолчанию скрипт подключается к:

```text
localhost:2000
```

---

## Настройки

Все настройки находятся внутри функции `main()` в файле:

```text
generate_carla_reid_dataset.py
```

Основные параметры:

```python
host = "localhost"
port = 2000
town = "Town10HD_Opt"
output_dir = "generated_dataset"

image_width = 1920
image_height = 1080

num_train_identities = 200
min_train_images_per_identity = 15
max_train_images_per_identity = 30

num_test_identities = 30
num_test_query_per_identity = 3
num_test_gallery_per_identity = 10
```
---

## Результат работы

После успешной генерации создаётся папка:

```text
generated_dataset/
├── images/
│   ├── train_000001_0000.jpg
│   ├── train_000001_0001.jpg
│   ├── test_query_100001_0000.jpg
│   ├── test_gallery_100001_0000.jpg
│   └── ...
├── train.csv
├── test_query.csv
├── test_gallery.csv
└── meta/
    ├── dataset_info.json
    ├── identities_appearance.json
    └── test_ground_truth.json
```

### Формат `train.csv`

```csv
image_id,x,y,w,h,vehicle_id
```

### Формат `test_query.csv`

```csv
image_id,x,y,w,h
```

### Формат `test_gallery.csv`

```csv
image_id,x,y,w,h
```
---

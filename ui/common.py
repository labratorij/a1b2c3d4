import os
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SERVICE_DIR = os.environ.get("SERVICE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("UI_DATA_ROOT", os.path.join(SERVICE_DIR, "data", "dataset"))
IMAGES_DIR = os.environ.get("UI_IMAGES_DIR", os.path.join(DATA_ROOT, "images"))


def api_get(path: str, timeout: int = 120):
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, timeout: int = 3600, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)


def inspect_dir(path: str):
    try:
        r = requests.get(f"{API_URL}/data/inspect", params={"path": path}, timeout=120)
    except requests.RequestException as e:
        return None, f"Backend недоступен ({API_URL}): {e}"
    if r.status_code != 200:
        try:
            return None, r.json().get("detail", r.text)
        except ValueError:
            return None, r.text
    return r.json(), None


ROLE_LABEL = {"query": "запросы", "gallery": "галерея", "train": "обучение", "other": "прочее"}


def upload_form(key: str):
    st.caption("Файлы отправляются на сервер и складываются в рабочий каталог: "
               "изображения в `images/`, CSV рядом. Можно загружать по частям — "
               "каждая загрузка дополняет каталог, если не отмечено «очистить».")
    try:
        existing = api_get("/data/uploads", timeout=30)
    except requests.RequestException:
        existing = []
    default_name = "upload"
    if existing:
        labels = ["— новый каталог —"] + [f"{w['name']} — {w['images']} снимков, {w['csvs']} CSV"
                                          for w in existing]
        pick = st.selectbox("Ранее загруженное", labels, key=f"{key}_pick")
        if pick != labels[0]:
            w = existing[labels.index(pick) - 1]
            default_name = w["name"]
            st.session_state[f"{key}_uploaded"] = w["path"]
            if st.button(f"Удалить «{w['name']}» с сервера", key=f"{key}_del"):
                requests.delete(f"{API_URL}/data/uploads/{w['name']}", timeout=60)
                st.session_state.pop(f"{key}_uploaded", None)
                st.rerun()

    name = st.text_input("Имя рабочего каталога", value=default_name, key=f"{key}_ws")
    images = st.file_uploader("Изображения", type=["jpg", "jpeg", "png", "bmp", "webp"],
                              accept_multiple_files=True, key=f"{key}_imgs")
    csvs = st.file_uploader("CSV с аннотациями", type=["csv"], accept_multiple_files=True,
                            key=f"{key}_csvs")
    archive = st.file_uploader("Либо ZIP-архив каталога целиком", type=["zip"],
                               accept_multiple_files=True, key=f"{key}_zip",
                               help="Удобнее для больших наборов: браузер отдаёт один файл "
                                    "вместо тысяч. Внутренняя структура значения не имеет — "
                                    "файлы раскладываются по типу.")
    reset = st.checkbox("Очистить каталог перед загрузкой", value=False, key=f"{key}_reset")

    picked = list(images or []) + list(csvs or []) + list(archive or [])
    total = sum(f.size for f in picked)
    if picked:
        st.caption(f"К отправке: {len(picked)} файлов, {total / 2 ** 20:.1f} МБ")
    if st.button("Загрузить на сервер", key=f"{key}_go", type="secondary", disabled=not picked):
        payload = [("files", (f.name, f.getvalue())) for f in picked]
        with st.spinner(f"Отправка {len(picked)} файлов…"):
            r = api_post("/data/upload", files=payload,
                         data={"name": name, "reset": str(bool(reset)).lower()})
        if r.status_code != 200:
            st.error(f"Ошибка {r.status_code}: {r.json().get('detail', r.text)}")
        else:
            d = r.json()
            st.session_state[f"{key}_uploaded"] = d["data_dir"]
            st.success(f"Принято файлов: {d['received']}")
            for s in d["skipped"]:
                st.warning(s)
            st.rerun()
    return st.session_state.get(f"{key}_uploaded")


def dataset_picker(key: str, roles, default: str = DATA_ROOT):
    source = st.radio("Источник данных", ["Каталог на сервере", "Загрузить файлы"],
                      horizontal=True, key=f"{key}_src")
    if source == "Загрузить файлы":
        path = upload_form(key)
        if not path:
            st.info("Выберите файлы и нажмите «Загрузить на сервер».")
            return None, None
        st.caption(f"Рабочий каталог: `{path}`")
    else:
        path = st.text_input(
            "Каталог с данными", value=default, key=f"{key}_dir",
            help="Путь на стороне сервиса. Внутри сами находятся изображения и CSV — "
                 "указывать их по отдельности не нужно.")
    report, err = inspect_dir(path)
    if err:
        st.error(err)
        return path, None
    if report["problems"]:
        for p in report["problems"]:
            st.error(p)
        return path, None

    st.caption(f"Изображения: `{report['images_dir']}` — {report['images']:,} файлов"
               .replace(",", " "))
    usable = [c for c in report["csvs"] if c["usable"]]
    if usable:
        with st.expander(f"Найдено CSV: {len(usable)}", expanded=False):
            st.dataframe(pd.DataFrame([{
                "файл": c["name"],
                "роль": ROLE_LABEL.get(c["role"], c["role"]),
                "строк": c["rows"],
                "разметка vehicle_id": "есть" if c["has_vehicle_id"] else "нет",
            } for c in usable]), width="stretch", hide_index=True)
    for c in report["csvs"]:
        if not c["usable"]:
            st.warning(f"{c['name']}: {c['problem']}")

    chosen = {}
    cols = st.columns(len(roles))
    for col, (field, label, suggest_key) in zip(cols, roles):
        names = [c["name"] for c in usable]
        suggested = report["suggested"].get(suggest_key)
        default_name = os.path.basename(suggested) if suggested else None
        idx = names.index(default_name) if default_name in names else 0
        picked = col.selectbox(label, names, index=idx, key=f"{key}_{field}") if names else None
        chosen[field] = next((c["path"] for c in usable if c["name"] == picked), None)
    return path, chosen


def gallery_stats():
    try:
        return api_get("/gallery/stats", timeout=60)
    except requests.RequestException as e:
        st.error(f"Backend недоступен ({API_URL}): {e}")
        st.caption("Запустите backend: `python service/run_local.py` или `docker compose up`.")
        st.stop()


def sidebar_status():
    with st.sidebar:
        st.divider()
        st.caption(f"API: `{API_URL}`")
        try:
            h = api_get("/health")
        except requests.RequestException as e:
            st.error("Backend недоступен")
            st.caption(str(e)[:120])
            return None
        m, g = h["model"], h["gallery"]
        st.success("Backend доступен")
        st.metric("Объектов в галерее", f"{g['items']:,}".replace(",", " "))
        st.caption(f"{m['device']} · вход {m['input_size'][0]}×{m['input_size'][1]} · "
                   f"эмбеддинг {m['embedding_dim']}-d\n\nхранилище: {g['backend']}")
        return h

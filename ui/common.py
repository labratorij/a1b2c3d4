"""Общее для страниц интерфейса: адрес API и тонкие обёртки над вызовами."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_ROOT = os.environ.get("UI_DATA_ROOT", os.path.join("..", "..", "Датасет"))
IMAGES_DIR = os.environ.get("UI_IMAGES_DIR", os.path.join(DATA_ROOT, "images"))
SERVICE_DIR = os.environ.get("SERVICE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def api_get(path: str, timeout: int = 120):
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, timeout: int = 3600, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)


def gallery_stats():
    try:
        return api_get("/gallery/stats", timeout=60)
    except requests.RequestException as e:
        st.error(f"Backend недоступен ({API_URL}): {e}")
        st.caption("Запустите backend: `python service/run_local.py` или `docker compose up`.")
        st.stop()


def sidebar_status():
    """Блок состояния сервиса — одинаковый на всех страницах."""
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

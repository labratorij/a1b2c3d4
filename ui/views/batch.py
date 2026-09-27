"""Страница: пакетная обработка набора данных -> артефакты в формате сдачи."""
import os

import pandas as pd
import requests
import streamlit as st

from common import DATA_ROOT, api_post, sidebar_status


st.title("Пакетная обработка набора данных")
st.caption("Прогон всех запросов по галерее и формирование артефактов: submission.csv, "
           "candidates.csv, embeddings.npy.")
sidebar_status()

with st.form("batch"):
    data_dir = st.text_input(
        "Каталог с данными", value=DATA_ROOT,
        help="Путь на стороне сервиса (в контейнере он смонтирован из docker-compose). "
             "Внутри ожидаются images/ и CSV с аннотациями.")
    c1, c2 = st.columns(2)
    query_csv = c1.text_input("CSV запросов", value="test_query.csv",
                              help="Колонки image_id,x,y,w,h — с заголовком или без")
    gallery_csv = c2.text_input("CSV галереи", value="test_gallery.csv")
    c3, c4, c5 = st.columns(3)
    top_k = c3.number_input("Кандидатов на запрос (top-K)", 1, 100, 10)
    threshold = c4.slider("Порог режима отказа", 0.0, 1.0, 0.30, 0.01,
                          help="Ниже порога запрос получает пустой ответ в candidates.csv")
    rerank = c5.checkbox("k-reciprocal re-ranking", value=True,
                         help="Работает только на пакете запросов: +3.4 pt mAP@10. "
                              "В онлайн-поиске по одному фото не применяется.")
    out_dir = st.text_input("Каталог для результатов", value="outputs/service",
                            help="Относительно корня решения на стороне сервиса")
    submitted = st.form_submit_button("Запустить обработку", type="primary")

if submitted:
    payload = {
        "query_csv": os.path.join(data_dir, query_csv),
        "gallery_csv": os.path.join(data_dir, gallery_csv),
        "images_dir": os.path.join(data_dir, "images"),
        "output_dir": out_dir,
        "top_k": int(top_k),
        "threshold": float(threshold),
        "use_reranking": bool(rerank),
    }
    with st.spinner("Извлечение признаков и ранжирование — это может занять несколько минут…"):
        try:
            r = api_post("/jobs/batch", json=payload)
        except requests.RequestException as e:
            st.error(f"Backend недоступен: {e}")
            st.stop()
    if r.status_code != 200:
        st.error(f"Ошибка {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    st.session_state["batch_result"] = r.json()

res = st.session_state.get("batch_result")
if res:
    st.success("Обработка завершена")
    m = st.columns(5)
    m[0].metric("Запросов", res["n_query"])
    m[1].metric("Объектов галереи", res["n_gallery"])
    m[2].metric("С ответом", res["accepted"])
    m[3].metric("Отказов", res["n_query"] - res["accepted"])
    m[4].metric("Время", f"{res['seconds']:.0f} с")
    st.caption(f"Файлы записаны в `{res['output_dir']}` на стороне сервиса.")

    st.subheader("Результаты")
    for label, key, mime in (("submission.csv", "submission", "text/csv"),
                             ("candidates.csv", "candidates", "text/csv"),
                             ("embeddings.npy", "embeddings", "application/octet-stream")):
        path = res["files"][key]
        cols = st.columns([3, 1])
        if os.path.exists(path):                      # UI и API на одной машине - отдаём файл на скачивание
            size = os.path.getsize(path) / 2 ** 20
            cols[0].write(f"**{label}** — {size:.1f} МБ")
            with open(path, "rb") as f:
                cols[1].download_button("Скачать", f.read(), file_name=label, mime=mime, key=f"dl_{key}")
        else:
            cols[0].write(f"**{label}** — `{path}`")
            cols[1].caption("файл на стороне API")

    sub = res["files"]["submission"]
    if os.path.exists(sub):
        st.subheader("Предпросмотр ранжирования")
        df = pd.read_csv(sub, header=None, nrows=15)
        df.columns = ["query_id"] + [f"кандидат {i}" for i in range(1, len(df.columns))]
        st.dataframe(df, width="stretch", hide_index=True)
    cand = res["files"]["candidates"]
    if os.path.exists(cand):
        st.subheader("Принятые кандидаты (режим отказа)")
        cdf = pd.read_csv(cand)
        st.caption(f"Принято {len(cdf)} из {res['n_query']} запросов; для остальных ответ пустой — "
                   f"уверенность ниже порога.")
        if len(cdf):
            st.dataframe(cdf.head(15), width="stretch", hide_index=True)
            st.bar_chart(cdf.confidence.value_counts(bins=20).sort_index(), x_label="уверенность",
                         y_label="запросов")
else:
    st.info("Заполните форму и нажмите «Запустить обработку».")

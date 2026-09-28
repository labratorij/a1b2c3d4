import io
import os

import pandas as pd
import requests
import streamlit as st

from common import (API_URL, IMAGES_DIR, api_post, dataset_picker, gallery_stats, inspect_dir,
                    sidebar_status)
from PIL import Image, ImageDraw


st.title("Поиск транспортного средства по снимку")
sidebar_status()


st.header("База галереи")
s = gallery_stats()
c = st.columns(4)
c[0].metric("Объектов", f"{s['items']:,}".replace(",", " "))
c[1].metric("Уникальных ТС", s.get("vehicles") or "—")
c[2].metric("Размерность", f"{s['dim']}-d" if s.get("dim") else "—")
c[3].metric("Хранилище", s["backend"])

st.caption(f"База: {s['location']}. Индексация нужна один раз — записи сохраняются в СУБД "
           f"и переживают перезапуск сервиса.")

with st.expander("Загрузить или обновить базу", expanded=s["items"] == 0):
    data_dir, chosen = dataset_picker("bulk", [("csv_path", "CSV галереи", "bulk_csv")])
    st.caption("Колонки image_id,x,y,w,h[,vehicle_id]. vehicle_id не обязателен, "
               "но с ним результаты поиска понятнее.")
    with st.form("bulk"):
        col = st.columns(3)
        replace = col[0].checkbox("Очистить базу перед загрузкой", value=s["items"] == 0,
                                  help="Без очистки повторная загрузка того же CSV просто обновит записи")
        limit = col[1].number_input("Ограничить числом строк (0 = все)", 0, 100000, 0, step=100)
        fit = col[2].checkbox("Обучить PCA-проекцию", value=not s["projection"],
                              help="Нужно при первой загрузке. Проекция задаёт пространство базы, "
                                   "поэтому на непустой базе пересчитать её нельзя — только вместе с очисткой.")
        if st.form_submit_button("Загрузить базу", type="primary", disabled=chosen is None):
            payload = {"data_dir": data_dir, "csv_path": chosen["csv_path"],
                       "replace": bool(replace), "fit_projection": bool(fit),
                       "limit": int(limit) or None}
            with st.spinner("Индексация галереи…"):
                r = api_post("/gallery/bulk", json=payload)
            if r.status_code == 200:
                d = r.json()
                st.success(f"Проиндексировано {d['indexed']} объектов за {d['seconds']:.0f} с "
                           f"(не прочитано: {d['failed']}), размерность {d['dim']}-d")
                st.rerun()
            else:
                st.error(f"Ошибка {r.status_code}: {r.json().get('detail', r.text)}")
    if s["items"]:
        if st.button("Очистить базу"):
            requests.delete(f"{API_URL}/gallery", timeout=120)
            st.rerun()

if s["items"] == 0:
    st.warning("База пуста — загрузите галерею, иначе поиск невозможен.")
    st.stop()

st.header("Запрос")
left, right = st.columns([1, 1])
with left:
    up = st.file_uploader("Снимок для поиска", type=["jpg", "jpeg", "png", "bmp", "webp"])
    st.caption("Либо укажите image_id из каталога данных:")
    by_id = st.text_input("image_id из каталога", value="", placeholder="например 784b85da77984d3288f33380462144c6")
with right:
    st.write("**Рамка транспортного средства** (необязательно; без неё берётся весь кадр)")
    b = st.columns(4)
    bx = b[0].number_input("x", value=0, step=10)
    by = b[1].number_input("y", value=0, step=10)
    bw = b[2].number_input("w", value=0, step=10)
    bh = b[3].number_input("h", value=0, step=10)
    top_k = st.slider("Сколько кандидатов показать", 1, 20, 10)
    thr = st.slider("Порог режима отказа", 0.0, 1.0, 0.30, 0.01,
                    help="Ниже порога сервис отвечает отказом. 0.30 — максимум F1, "
                         "0.45 — точность 91%, 0.60 — точность 100% при полноте 39%")

report, _ = inspect_dir(data_dir)
images_dir = report["images_dir"] if report and report["images_dir"] else IMAGES_DIR

img_bytes, img_name = None, None
if up is not None:
    img_bytes, img_name = up.getvalue(), up.name
elif by_id.strip():
    path = os.path.join(images_dir, f"{by_id.strip()}.jpg")
    if os.path.exists(path):
        img_bytes, img_name = open(path, "rb").read(), os.path.basename(path)
        if bw == 0 and bh == 0:
            for c in sorted((report or {}).get("csvs", []),
                            key=lambda c: (c["role"] not in ("query", "gallery"), c["name"])):
                if not c["usable"]:
                    continue
                row = pd.read_csv(c["path"], encoding="utf-8-sig")
                row = row[row.image_id == by_id.strip()] if "image_id" in row.columns else row.head(0)
                if len(row):
                    r0 = row.iloc[0]
                    bx, by_, bw, bh = int(r0.x), int(r0.y), int(r0.w), int(r0.h)
                    by = by_
                    st.info(f"Рамка взята из {c['name']}: x={bx}, y={by}, w={bw}, h={bh}")
                    break
    else:
        st.error(f"Файл не найден: {path}")

if img_bytes and st.button("Искать", type="primary"):
    data = {"top_k": int(top_k), "threshold": float(thr)}
    if bw > 0 and bh > 0:
        data["bbox"] = f"{bx},{by},{bw},{bh}"
    with st.spinner("Формирование признака и поиск…"):
        r = api_post("/search", files={"file": (img_name, img_bytes)}, data=data, timeout=600)
    if r.status_code != 200:
        st.error(f"Ошибка {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    st.session_state["search"] = (r.json(), img_bytes, (bx, by, bw, bh) if bw > 0 and bh > 0 else None)

if "search" in st.session_state:
    res, qbytes, qbox = st.session_state["search"]
    st.header("Результат")
    qcol, rcol = st.columns([1, 3])
    with qcol:
        qimg = Image.open(io.BytesIO(qbytes)).convert("RGB")
        if qbox:
            d = ImageDraw.Draw(qimg)
            x, y, w, h = qbox
            d.rectangle([x, y, x + w, y + h], outline=(0, 200, 83), width=max(3, qimg.width // 200))
        st.image(qimg, caption="Запрос", width="stretch")
        if res["matched"]:
            st.success(f"**Найдено: ТС {res['vehicle_id'] or '—'}**\n\n"
                       f"уверенность {res['best_score']:.3f} ≥ порог {res['threshold']:.2f}")
        else:
            st.warning(f"**Отказ — уверенного совпадения нет**\n\n{res['refusal_reason']}")
        st.caption(f"Поиск по {res['gallery_size']:,} объектам базы".replace(",", " "))

    with rcol:
        st.subheader(f"Кандидаты (топ-{len(res['candidates'])})")
        rows = []
        cols = st.columns(5)
        for i, cand in enumerate(res["candidates"]):
            path = os.path.join(images_dir, f"{cand['image_id']}.jpg")
            rows.append({"#": i + 1, "image_id": cand["image_id"], "vehicle_id": cand["vehicle_id"],
                         "уверенность": round(cand["score"], 4), "принят": "да" if cand["accepted"] else "нет"})
            with cols[i % 5]:
                if os.path.exists(path):
                    im = Image.open(path).convert("RGB")
                    color = (0, 200, 83) if cand["accepted"] else (160, 160, 160)
                    bw_px = max(4, int(min(im.size) * 0.03)) if cand["accepted"] else 2
                    from PIL import ImageOps
                    im = ImageOps.expand(im, border=bw_px, fill=color)
                    st.image(im, width="stretch",
                             caption=f"#{i+1} ТС {cand['vehicle_id'] or '—'}\n"
                                     f"{cand['score']:.3f}")
                else:
                    st.write(f"#{i+1} `{cand['image_id'][:10]}…`  \nсходство {cand['score']:.3f}")
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button("Скачать результат (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="search_result.csv", mime="text/csv")

    if res["candidates"]:
        st.subheader("Почему модель так решила")
        st.caption("Grad-CAM: тёплым выделены области, из-за которых снимки признаны похожими. "
                   "Сходство симметрично, поэтому карта строится для обоих снимков — так видно, "
                   "смотрела ли модель на соответственные части. Подложка обесцвечена, чтобы "
                   "цвет означал важность, а не окраску кузова.")
        opts = {f"#{i+1} — ТС {c['vehicle_id'] or '—'} ({c['score']:.3f})": c["image_id"]
                for i, c in enumerate(res["candidates"])}
        col = st.columns([2, 1, 1])
        choice = col[0].selectbox("С каким кандидатом сравнивать", list(opts), index=0)
        mode = col[1].selectbox(
            "Вид", ["pair", "side_by_side", "overlay"],
            format_func=lambda v: {"pair": "обе карты: запрос и кандидат",
                                   "side_by_side": "только запрос: кроп и карта",
                                   "overlay": "только запрос: карта"}[v])
        if col[2].button("Построить карту"):
            with st.spinner("Считаю градиенты…"):
                data = {"reference_image_id": opts[choice], "mode": mode}
                if qbox:
                    data["bbox"] = ",".join(str(int(v)) for v in qbox)
                rr = api_post("/explain", files={"file": ("q.jpg", qbytes)}, data=data, timeout=900)
            if rr.status_code != 200:
                st.error(f"Ошибка {rr.status_code}: {rr.json().get('detail', rr.text)}")
            else:
                st.session_state["explain"] = (rr.content, dict(rr.headers))

    if "explain" in st.session_state:
        png, hdr = st.session_state["explain"]
        st.image(png, width="stretch")
        cols = st.columns(4)
        cols[0].metric("Моделей в карте", hdr.get("x-members-used", "—"))
        cols[1].metric("Сходство пары", f"{float(hdr.get('x-score', 0)):.3f}" if hdr.get("x-score") else "—")
        cols[2].metric("Важность в центре: запрос", f"{float(hdr.get('x-focus-center-share', 0)):.0%}")
        ref_share = hdr.get("x-focus-center-share-reference")
        cols[3].metric("Важность в центре: кандидат", f"{float(ref_share):.0%}" if ref_share else "—")
        st.caption("Верхняя строка — запрос, нижняя — кандидат; слева кроп, справа карта. "
                   "«Важность в центре» — доля карты внутри центральных 70% кропа: кроп сделан "
                   "по рамке ТС, поэтому низкое значение означало бы, что модель цепляется за фон.")
        st.download_button("Скачать карту (PNG)", png, file_name="gradcam.png", mime="image/png")

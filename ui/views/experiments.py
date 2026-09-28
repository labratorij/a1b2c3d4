import base64
import os
import re

import streamlit as st

from common import SERVICE_DIR, sidebar_status

DOC = os.environ.get("EXPERIMENTS_MD") or os.path.join(SERVICE_DIR, "EXPERIMENTS.md")
DOC_ROOT = os.path.dirname(os.path.abspath(DOC))

st.title("Журнал экспериментов")
st.caption("Все проверенные подходы, их результаты и выводы — включая отрицательные, "
           "чтобы их не повторяли.")
sidebar_status()

if not os.path.exists(DOC):
    st.error(f"Файл не найден: {DOC}")
    st.stop()

text = open(DOC, encoding="utf-8").read()

with st.sidebar:
    st.header("Разделы")
    for line in text.split("\n"):
        if line.startswith("## "):
            st.markdown(f"- {line[3:]}")

IMG = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*$")


def render_svg(path: str, alt: str) -> None:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" alt="{alt}" '
                f'style="width:100%;max-width:820px;display:block;margin:0.5rem 0;"/>',
                unsafe_allow_html=True)


buf = []
for line in text.split("\n"):
    m = IMG.match(line)
    if not m:
        buf.append(line)
        continue
    if buf:
        st.markdown("\n".join(buf))
        buf = []
    src = m.group("src")
    path = src if os.path.isabs(src) else os.path.join(DOC_ROOT, src)
    if path.lower().endswith(".svg") and os.path.exists(path):
        render_svg(path, m.group("alt"))
    elif os.path.exists(path):
        st.image(path, caption=m.group("alt"))
    else:
        st.caption(f"(график не найден: {src})")
if buf:
    st.markdown("\n".join(buf))

st.divider()
with open(DOC, "rb") as f:
    st.download_button("Скачать EXPERIMENTS.md", f.read(), file_name="EXPERIMENTS.md", mime="text/markdown")

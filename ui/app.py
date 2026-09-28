import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   

st.set_page_config(page_title="Vehicle ReID", layout="wide")

VIEWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")

nav = st.navigation([
    st.Page(os.path.join(VIEWS, "home.py"), title="Как пользоваться", default=True),
    st.Page(os.path.join(VIEWS, "batch.py"), title="Пакетная обработка"),
    st.Page(os.path.join(VIEWS, "search.py"), title="Поиск по фото"),
    st.Page(os.path.join(VIEWS, "experiments.py"), title="Эксперименты"),
])
nav.run()

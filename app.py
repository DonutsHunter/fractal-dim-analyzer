import streamlit as st

import scripts.lib as sl

st.set_page_config(page_title="Фрактальная размерность 📈", page_icon="📈")

st.markdown("<h1 style='text-align: center;'>Анализ размерности фрактальных структур</h1>", unsafe_allow_html=True)
st.write(
    """"""
)

sl.fractal_dim()
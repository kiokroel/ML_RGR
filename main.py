from pages import *
import streamlit as st

pages = {
    'Главная': main_page,
    'О наборе данных': about_df_page,
    'Визуализация': visualization_page,
    'Предсказание': predict_page,
}

page_name = st.sidebar.radio('Выберите страницу', pages.keys())

pages[page_name]()

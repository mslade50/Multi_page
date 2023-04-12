import streamlit as st
from pages import daily_signals, indicies, positions, single_names

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def daily_signals_page():
    st.sidebar.markdown("# Daily Signals")
    daily_signals.daily_signals_app()

def indicies_page():
    st.sidebar.markdown("# Indices")
    indicies.indicies_app()

def positions_page():
    st.sidebar.markdown("# Positions")
    positions.positions_app()

def single_names_page():
    st.sidebar.markdown("# Single Names")
    single_names.single_names_app()

page_names_to_funcs = {
    "Daily Signals": daily_signals_page,
    "Indices": indicies_page,
    "Positions": positions_page,
    "Single Names": single_names_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

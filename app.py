import streamlit as st
from pages import daily_signals, indicies, positions, single_names, currency_pairs, commodities, user_input, variance_sim

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

def currency_pairs_page():
    st.sidebar.markdown("# Currency Pairs")
    currency_pairs.currency_pairs_app()

def commodities_page():
    st.sidebar.markdown("# Commodities")
    commodities.commodities_app()

def user_input_page():
    st.sidebar.markdown("# User Input")
    user_input.user_input_app()

def variance_sim_page():
    st.sidebar.markdown("# Variance Simulation")
    variance_sim.variance_sim_app()

page_names_to_funcs = {
    "Daily Signals": daily_signals_page,
    "Indices": indicies_page,
    "Positions": positions_page,
    "Single Names": single_names_page,
    "Currency Pairs": currency_pairs_page,
    "Commodities": commodities_page,
    "User Input": user_input_page,
    "Variance Simulation": variance_sim_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

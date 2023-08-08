import streamlit as st
from pages import indicies, single_names, currency_pairs, commodities, user_input, variance_sim, trade_sizing

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def indicies_page():
    st.sidebar.markdown("# Indices")
    indicies.indicies_app()

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

def trade_sizing_page():
    st.sidebar.markdown("# Trade Sizing")
    trade_sizing.trade_sizing_app()

def fwd_distributions_page():
    st.sidebar.markdown("# Forward Distributions")
    fwd_distributions.fwd_distributions_app()

page_names_to_funcs = {
    "Indices": indicies_page,
    "Single Names": single_names_page,
    "Currency Pairs": currency_pairs_page,
    "Commodities": commodities_page,
    "User Input": user_input_page,
    "Variance Simulation": variance_sim_page,
    "Trade Sizing": trade_sizing_page,
    "Forward Distributions": fwd_distributions_page  # added this line
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

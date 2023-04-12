import streamlit as st
from daily_signals import daily_signals_app
from indicies import indicies_app
from positions import positions_app
from single_names import single_names_app

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def main():
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.selectbox(
        "Choose a page:",
        options=[
            "Daily Signals",
            "Indices",
            "Positions",
            "Single Names",
        ],
    )

    if app_selection == "Daily Signals":
        daily_signals_app()
    elif app_selection == "Indices":
        indicies_app()
    elif app_selection == "Positions":
        positions_app()
    elif app_selection == "Single Names":
        single_names_app()

if __name__ == "__main__":
    main()

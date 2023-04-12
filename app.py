import streamlit as st
from pages import daily_signals, indicies, positions, single_names

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
        daily_signals
    elif app_selection == "Indices":
        indicies
    elif app_selection == "Positions":
        positions
    elif app_selection == "Single Names":
        single_names

if __name__ == "__main__":
    main()

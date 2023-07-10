# pages/monte_carlo.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def monte_carlo_sim(win_prob, win_loss_ratio, start_capital, num_trials, num_paths, bet_sizing, stake_type):
    paths = []
    negative_endings = 0
    for _ in range(num_paths):
        capital = start_capital
        capital_path = [0]  # Start at 0 for PnL
        for _ in range(num_trials):
            if stake_type == 'flat':
                bet_size = bet_sizing * start_capital  # Flat stake
            else:
                bet_size = bet_sizing * capital  # Variable stake
            outcome = np.random.choice(['win', 'loss'], p=[win_prob, 1-win_prob])
            if outcome == 'win':
                capital += bet_size * win_loss_ratio
            else:
                capital -= bet_size
            capital_path.append(capital - start_capital)  # Subtract start capital for PnL
        paths.append(capital_path)
        if capital_path[-1] < 0:  # Check if final value of path is negative
            negative_endings += 1
    return paths, negative_endings

def monte_carlo_app():
    st.markdown("# Monte Carlo Simulation")

    win_prob = st.sidebar.number_input("Winning probability", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    win_loss_ratio = st.sidebar.number_input("Win/Loss Ratio", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
    start_capital = st.sidebar.number_input("Starting Capital", min_value=0.0, max_value=1000000.0, value=100000.0, step=100.0)
    num_trials = st.sidebar.number_input("Number of trials", min_value=0, max_value=10000, value=1000, step=100)
    bet_sizing = st.sidebar.number_input("Bet Sizing", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    num_paths = st.sidebar.number_input("Number of paths", min_value=0, max_value=10000, value=50, step=5)
    stake_type = st.sidebar.selectbox("Stake Type", ('flat', 'variable'))

    if st.button("Run Simulation"):
        paths, negative_endings = monte_carlo_sim(win_prob, win_loss_ratio, start_capital, num_trials, num_paths, bet_sizing, stake_type)

        # Calculate the initial expected value (EV)
        initial_bet_size = bet_sizing * start_capital if stake_type == 'flat' else bet_sizing * start_capital
        EV = ((win_prob * initial_bet_size * win_loss_ratio) - ((1 - win_prob) * initial_bet_size))/initial_bet_size*100
        st.write(f"Initial Expected Value (EV): {EV}%")

        # Calculate the Kelly criterion bet size and round to 2 decimal places
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fraction = round(kelly_fraction, 2)  # round to 2 decimal places
        st.write(f"Kelly criterion bet size (% of capital): {kelly_fraction * 100}%")

        # Calculate the average percentage of paths ending with negative PnL
        average_neg_endings = np.mean(negative_endings)
        st.write(f"Avg % of paths ending with negative PnL: {average_neg_endings/num_paths*100}%")

        # Calculate the realized EV per trade
        realized_EVs = [path[-1] for path in paths]  # Profit or loss at the end of each path
        average_realized_EV = np.mean(realized_EVs) / num_trials  # Average profit or loss per trial
        st.write(f"Realized Expected Value (EV) per trade: {average_realized_EV}")
        
        # Create DataFrame for Plotly
        df = pd.DataFrame(paths).T
        df.index.name = "Trial"
        df.columns.name = "Path"
        
        fig = go.Figure()
        
        for path in df.columns:
            fig.add_trace(go.Scatter(y=df[path], mode='lines', name=f'Path {path}'))

        fig.update_layout(height=600, width=800, title_text="Monte Carlo Simulation Paths")
        st.plotly_chart(fig)


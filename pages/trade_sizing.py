import streamlit as st
import yfinance as yf
from ta.volatility import AverageTrueRange

# Function to calculate number of shares to buy
def calculate_trade_size(account_size, risk_allocation, entry_level, stop_level, base_currency, target_currency):
    risk_per_trade = account_size * risk_allocation / 100  # Converting percentage risk allocation to absolute value
    risk_per_unit = abs(entry_level - stop_level)

    # Convert risk_per_trade to target currency if necessary
    if target_currency != base_currency:
        forex_ticker = f"{base_currency}{target_currency}=X"
        exchange_rate = yf.Ticker(forex_ticker).info['regularMarketPrice']
        risk_per_trade = risk_per_trade / exchange_rate

    trade_size = risk_per_trade / risk_per_unit
    return int(trade_size)

# Function to calculate ATR-based stop-loss
def calculate_atr_stop_level(ticker, entry_level, atr_multiplier=2):
    data = yf.download(ticker, period='1y')
    atr = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range().iloc[-1]
    stop_level = entry_level - atr * atr_multiplier
    return stop_level

# Main function
def trade_sizing_app():
    st.markdown("# Trade Sizing App")
    ticker = st.text_input("Stock Ticker")
    entry_level = st.number_input("Trade Entry Level", min_value=0.0, value=100.0, step=0.01)
    trade_direction = st.selectbox("Trade Direction", ['Long', 'Short'])
    account_size = st.number_input("Account Size", min_value=0.0, value=10000.0, step=100.0)
    risk_allocation = st.number_input("Target Risk as % of Account", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    stop_loss_override = st.number_input("Stop Loss Override (Leave blank for ATR-based stop loss)", min_value=0.0, value=0.0, step=0.01)
    base_currency = st.text_input("Base Currency", value="USD")
    target_currency = st.text_input("Target Currency", value="USD")

    if st.button("Calculate Trade Size"):
        # Determine stop level
        if stop_loss_override == 0.0:
            stop_level = calculate_atr_stop_level(ticker, entry_level)
        else:
            stop_level = stop_loss_override

        trade_size = calculate_trade_size(account_size, risk_allocation, entry_level, stop_level, base_currency, target_currency)
        
        dollar_risk = trade_size * abs(entry_level - stop_level)

        st.write(f"Shares: {trade_size}")
        st.write(f"Entry: {entry_level}")
        st.write(f"Stop Level: {stop_level}")
        st.write(f"$ Risk: {dollar_risk}")

# Call the function
trade_sizing_app()


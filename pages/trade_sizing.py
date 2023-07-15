import streamlit as st
import yfinance as yf
import ta

def calculate_trade_size(account_size, risk_allocation, entry_level, stop_level, ticker, asset_class):
    risk_per_trade = account_size * risk_allocation / 100  # Converting percentage risk allocation to absolute value
    risk_per_unit = abs(entry_level - stop_level)

    if asset_class == 'Forex':
        # Break down the entered currency ticker
        base_currency, target_currency = ticker[:3], ticker[3:6]

        # If the last 3 letters before =X are not USD, we need to multiply the calculated shares by USD<last 3 letters>=X last closing price
        if target_currency != 'USD':
            forex_ticker = f"USD{target_currency}=X"
            forex_data = yf.download(forex_ticker, period="1d")['Close'].iloc[0]
            risk_per_trade = risk_per_trade / forex_data

        # We also need to check if the first three letters of the entered currency are USD. If they are also not USD then we need to do the exact same thing we did with the last 3 letters
        if base_currency != 'USD':
            forex_ticker = f"USD{base_currency}=X"
            forex_data = yf.download(forex_ticker, period="1d")['Close'].iloc[0]
            risk_per_trade = risk_per_trade / forex_data

    trade_size = risk_per_trade / risk_per_unit
    return int(trade_size)

def trade_sizing_app():
    st.markdown("# Trade Sizing")

    ticker = st.text_input("Enter the Ticker")
    entry_level = st.number_input("Enter the Entry Level", min_value=0.0, value=100.0, step=0.01)
    account_size = st.number_input("Enter the Account Size", min_value=0.0, value=100000.0, step=0.01)
    risk_allocation = st.number_input("Target Risk as % of Account", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    stop_loss_override = st.text_input("Stop Loss Override", value="")
    asset_class = st.selectbox("Asset Class", ['Stocks', 'Options', 'Futures', 'Forex'])
    direction = st.selectbox("Direction of Trade", ['Long', 'Short'])

    if ticker and st.button("Calculate Trade Size"):
        data = yf.download(ticker, period="1mo")  # Downloading last 30 days of data
        atr = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False).iloc[-1]

        if stop_loss_override:
            stop_level = float(stop_loss_override)
        else:
            if direction == 'Long':
                stop_level = entry_level - 2 * atr
            else:
                stop_level = entry_level + 2 * atr

        trade_size = calculate_trade_size(account_size, risk_allocation, entry_level, stop_level, ticker, asset_class)

        st.write(f"Shares: {trade_size}")
        st.write(f"Entry: {entry_level}")
        st.write(f"Stop Level: {stop_level}")
        st.write(f"$ Risk: {trade_size * risk_per_unit}")

trade_sizing_app()



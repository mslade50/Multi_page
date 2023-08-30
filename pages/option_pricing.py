import streamlit as st
from mdm_module_3 import fig_creation
from option_scrape import option_chain_prices
import datetime as dt
import numpy as np
import pandas_market_calendars as mcal
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import pandas as pd
import yfinance as yf

ticker = st.text_input('Enter the ticker symbol:', value='SPY')
if st.button('Get Prices'):
    if ticker == "^GSPC": 
        ticker_2 = "^SPX"
    else:
        ticker_2 = ticker
    def kelly_criterion(prob_win, payoff):
        odds = payoff - 1 if payoff > 0 else payoff + 1
        if abs(payoff) < 1e-5 or abs(odds) < 1:  # Checking if payoff is very close to zero or odds is less than abs 0.25
            return 0
        return (prob_win - ((1 - prob_win) / odds))/20
    
    def kde_option_price_and_prob(forward_prices, K, option_type="call"):
        kde = gaussian_kde(forward_prices)
        min_limit = np.min(forward_prices) * .75
        max_limit = np.max(forward_prices) * 1.25
    
        if option_type == "call":
            in_the_money_values = [price for price in forward_prices if price > K]
            prob_profit = len(in_the_money_values) / len(forward_prices)
            integrand = lambda x: np.maximum(x - K, 0) * kde(x)
            payoff = np.mean(in_the_money_values) - K if in_the_money_values else 0
        else:
            in_the_money_values = [price for price in forward_prices if price < K]
            prob_profit = len(in_the_money_values) / len(forward_prices)
            integrand = lambda x: np.maximum(K - x, 0) * kde(x)
            payoff = K - np.mean(in_the_money_values) if in_the_money_values else 0
    
        option_value, _ = quad(integrand, min_limit, max_limit)
    
        return option_value, prob_profit, payoff
    option_data = option_chain_prices(ticker_2, price_type="midpoint")
    available_expiries = option_data["Available_Expiries"]
    
    def trading_days_to_expiry(expiry_date):
        # Get NYSE market calendar
        nyse = mcal.get_calendar('NYSE')
    
        # Define the date range
        start_date = dt.datetime.today()
        end_date = expiry_date
    
        # Get trading schedule between the dates
        trading_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
        return len(trading_schedule)
    
    
    lookback_periods = [4, 10, 21, 63] 
    # lookback_periods = [4, 21, 63,252] 
    forward_prices_dfs = {}
    days_variables=[]
    closest_expiries=[]
    for lookback_period in lookback_periods:
        closest_expiry = None
        closest_days_to_expiry = None
    
        for expiry in available_expiries:  
            days_to_expiry = trading_days_to_expiry(expiry)-1
    
            # Check if this expiry is closer to the target lookback period
            if closest_days_to_expiry is None or abs(lookback_period - days_to_expiry) < abs(lookback_period - closest_days_to_expiry):
                closest_expiry = expiry
                closest_days_to_expiry = days_to_expiry
    
        days_variables.append(closest_days_to_expiry)
        closest_expiries.append(closest_expiry)  
    
        # Use the closest_days_to_expiry value to create forward distributions
    forward_prices_df= fig_creation(ticker, 12, (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d'), .13, days_variables, "no")
    print(days_variables)
    # forward_prices_df = fig_creation(ticker, 12, (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d'), .13, 21,"no")
    
    results = []
    # Assuming days_variables has three elements
    days_variable1, days_variable2, days_variable3,days_variable4 = days_variables
    filtered_options_data = [option for option in option_data["Options_Data"] if option["Expiry"] in closest_expiries]
    option_data["Options_Data"] = filtered_options_data
    # Loop through each option in the data
    for option in option_data["Options_Data"]:
        expiry = option["Expiry"]
        strike = option["Strike"]
        option_type = option["Type"].lower()
        
        # Get the appropriate forward_prices based on expiry 
        days_to_expiry = (expiry - dt.date.today()).days
        if days_to_expiry <= 7:
            column_name = f"Forward_{days_variable1}d_price"
        elif days_to_expiry <= 15:
            column_name = f"Forward_{days_variable2}d_price"
        elif days_to_expiry <= 40:
            column_name = f"Forward_{days_variable3}d_price"
        else:
            column_name = f"Forward_{days_variable4}d_price"
        
        forward_prices = forward_prices_df[column_name].values
        theoretical_price, prob_profit, payoff = kde_option_price_and_prob(forward_prices, strike, option_type)
        
        # Compare the theoretical_price with market price
        market_price = option["Market_Price"]
        difference = theoretical_price - market_price
        actual_payoff = (payoff - market_price) / market_price + 1 if market_price != 0 and payoff >= 0 else (payoff - 1) if market_price != 0 else 0
        kelly_bet_size = kelly_criterion(prob_profit, actual_payoff)    # Store in the results list
        results.append({
            "Expiry": expiry,
            "Type": option_type,
            "Strike": strike,
            "Mkt": market_price,
            "Theo": theoretical_price,
            "Diff": difference,
            "win%":prob_profit,
            "Exp":payoff,
            "Payoff":actual_payoff,
            "KC": kelly_bet_size
        })
    
    # Convert the results list to a DataFrame and sort by the absolute difference
    df = pd.DataFrame(results)
    
    df['R_disc'] = df['Diff'] / df['Mkt']
    df['Final_KC']=df.KC*(1+(df.R_disc/10))*100
    
    df_sorted = df.sort_values(by=["Final_KC", "Diff"], ascending=[False, False]).round(2)
    df_sorted.to_csv(f"{ticker}_options_pricing.csv", index=False)
    
    # Assume the current price is stored in this variable
    stock = yf.Ticker(ticker)
    current_price = stock.history().tail(1)["Close"].values[0]
    
    # Define a function to check if an option is out of the money
    def is_otm(row):
        if row['Type'] == 'call' and row['Strike'] > current_price:
            return True
        elif row['Type'] == 'put' and row['Strike'] < current_price:
            return True
        else:
            return False
    
    # Filter the DataFrame to include only the rows where the option is OTM
    df_otm = df_sorted[df_sorted.apply(is_otm, axis=1)]
    
    # Continue with your existing code using the df_otm DataFrame
    top_final_kc_per_date = df_otm.groupby('Expiry')['Final_KC'].nlargest(3).reset_index(level=0, drop=True)
    top_final_kc_per_date = df_otm.loc[top_final_kc_per_date.index].drop(columns=['R_disc', 'Exp', 'KC', 'Diff'])
    
    # Print the result
    st.dataframe(top_final_kc_per_date)

    

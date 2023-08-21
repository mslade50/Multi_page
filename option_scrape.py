import yfinance as yf
import datetime as dt

def option_chain_prices(ticker, price_type="midpoint"):
    stock = yf.Ticker(ticker)
    current_price = stock.history().tail(1)["Close"].values[0]  # Get the current price

    today = dt.date.today()
    desired_expiries = [today + dt.timedelta(days=7), today + dt.timedelta(weeks=4), today + dt.timedelta(weeks=12)]
    
    available_expiries = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in stock.options]
    # print("Available Expiries:")
    for expiry in available_expiries:
        days_until_expiry = (expiry - today).days
        # print(f"Expiry: {expiry}, Days until expiry: {days_until_expiry}")

    closest_expiries = [min(available_expiries, key=lambda x: abs(x - date)) for date in desired_expiries]
    # print(closest_expiries)
    all_data = []
    expiry_strike_data = {}

    for expiry in available_expiries:
        opt = stock.option_chain(date=expiry.strftime('%Y-%m-%d'))
        
        valid_call_strikes = []
        valid_put_strikes = []

        for _, row in opt.calls.iterrows():
            if price_type == "midpoint":
                price = (row["bid"] + row["ask"]) / 2 +.01
            else:
                price = row["lastPrice"]+.01

            if price > 0.05:
                valid_call_strikes.append(row["strike"])
                all_data.append({"Ticker": ticker, "Expiry": expiry, "Strike": row["strike"], "Type": "Call", "Market_Price": price})

        for _, row in opt.puts.iterrows():
            if price_type == "midpoint":
                price = (row["bid"] + row["ask"]) / 2 +.01
            else:
                price = row["lastPrice"] +.01

            if price > 0.05:
                valid_put_strikes.append(row["strike"])
                all_data.append({"Ticker": ticker, "Expiry": expiry, "Strike": row["strike"], "Type": "Put", "Market_Price": price})

        available_strikes = list(set(valid_call_strikes + valid_put_strikes))
        expiry_strike_data[expiry] = sorted(available_strikes)

    return {
        "Options_Data": all_data,
        "Expiry_Strikes": expiry_strike_data,
        "Available_Expiries": available_expiries  # Add this line to return the available expiries
    }


if __name__ == "__main__":
    pass
# # Example using midpoint
# result = option_chain_prices("uvxy", price_type="midpoint")
# print(result)

# # Example using lastPrice
# result = option_chain_prices("AAPL", price_type="lastPrice")
# print(result["Expiry_Strikes"])
    

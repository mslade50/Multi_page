import streamlit as st

# Function to calculate Kelly criterion
def calculate_kelly(payoff, win_prob, fractional_kelly):
    return (win_prob - (1 - win_prob) / payoff) / fractional_kelly

# Adding the Kelly Sizing page to the dashboard
def kelly_sizing_page():
    st.title("Kelly Sizing")

    # Input fields
    payoff = st.number_input("Payoff", min_value=0.0, step=0.01)
    win_prob = st.number_input("Win Percentage", min_value=0.0, max_value=1.0, step=0.01)
    fractional_kelly = st.number_input("Fractional Kelly (e.g., input 20 for 1/20th Kelly)", min_value=1)
    account_size = st.number_input("Account Size", min_value=0.0, step=0.01)

    if st.button("Calculate Bet Size"):
        if payoff > 0 and win_prob > 0 and win_prob <= 1 and fractional_kelly > 0 and account_size > 0:
            kelly_fraction = calculate_kelly(payoff, win_prob, fractional_kelly)
            dollar_bet = kelly_fraction * account_size
            st.success(f"Recommended Bet Size: {kelly_fraction * 100:.2f}% of your account")
            st.success(f"Recommended Bet Size in Dollars: ${dollar_bet:.2f}")
        else:
            st.error("Please ensure all inputs are filled out correctly.")

# Assuming you already have a main app that handles page selection
def main():
    # Other pages in the dashboard...
    
    # Add Kelly Sizing page
    kelly_sizing_page()

if __name__ == "__main__":
    main()

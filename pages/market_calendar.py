import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import calendar

# Reusable request function
def request(url: str, method: str = "get", timeout: int = 0, **kwargs) -> requests.Response:
    method = method.lower()
    if method not in ["delete", "get", "head", "patch", "post", "put"]:
        raise ValueError(f"Invalid method: {method}")
    headers = kwargs.pop("headers", {})
    headers["User-Agent"] = headers.get(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    )
    func = getattr(requests, method)
    return func(
        url,
        headers=headers,
        timeout=timeout or 10,
        **kwargs,
    )

# Function to get filter parameters for earnings request
def get_filters(date_str: str) -> str:
    return f"?filter[selected_date]={date_str}&filter[with_rating]=false&filter[currency]=USD"

# Class to fetch earnings events
class EventsFetcher:
    @staticmethod
    def get_next_earnings(limit: int = 5, start_date: date = date.today()) -> pd.DataFrame:
        base_url = "https://seekingalpha.com/api/v3/earnings_calendar/tickers"
        df_earnings = pd.DataFrame()

        for _ in range(0, limit):
            start_date = pd.to_datetime(start_date)
            date_str = str(start_date.strftime("%Y-%m-%d"))
            response = request(base_url + get_filters(date_str), timeout=10)
            json = response.json()
            try:
                data = json["data"]
                cleaned_data = [x["attributes"] for x in data]
                temp_df = pd.DataFrame.from_records(cleaned_data)
                temp_df = temp_df.drop(columns=["sector_id"], errors='ignore')
                temp_df["Date"] = start_date
                df_earnings = pd.concat([df_earnings, temp_df], ignore_index=True)
                start_date = start_date + timedelta(days=1)
            except KeyError:
                pass

        df_earnings = df_earnings.rename(
            columns={
                "slug": "Ticker",
                "name": "Name",
                "release_time": "Release Time",
                "exchange": "Exchange",
            }
        )

        return df_earnings

# Function to scrape Forex Factory calendar
def scrape_calendar(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    calendar_table = soup.find('table', class_='calendar__table')
    if not calendar_table:
        st.error(f"Could not find the calendar table on the page at {url}.")
        return None, None

    rows = calendar_table.find_all('tr', class_='calendar__row')
    events = []

    impact_map = {
        'icon--ff-impact-red': 'High Impact',
        'icon--ff-impact-ora': 'Medium Impact',
        'icon--ff-impact-yel': 'Low Impact',
        'icon--ff-impact-gra': 'Non-Economic'
    }

    current_date = None
    current_time = None

    for row in rows:
        date = row.find('td', class_='calendar__date')
        time = row.find('td', class_='calendar__time')
        currency = row.find('td', class_='calendar__currency')
        impact = row.find('td', class_='calendar__impact')
        event_name = row.find('td', class_='calendar__event')
        actual = row.find('td', class_='calendar__actual')
        forecast = row.find('td', class_='calendar__forecast')
        previous = row.find('td', class_='calendar__previous')

        if date and date.get_text(strip=True):
            current_date = date.get_text(strip=True)

        if time and time.get_text(strip=True):
            current_time = time.get_text(strip=True)

        if not current_time:
            current_time = 'N/A'

        if currency and impact and event_name:
            impact_span = impact.find('span')
            impact_class = impact_span['class'][1] if impact_span and len(impact_span['class']) > 1 else None
            impact_level = impact_map.get(impact_class, 'Unknown Impact')

            event = {
                'Date': current_date,
                'Time': current_time,
                'Currency': currency.get_text(strip=True),
                'Impact': impact_level,
                'Event': event_name.get_text(strip=True),
                'Actual': actual.get_text(strip=True) if actual else 'N/A',
                'Forecast': forecast.get_text(strip=True) if forecast else 'N/A',
                'Previous': previous.get_text(strip=True) if previous else 'N/A'
            }
            events.append(event)

    df = pd.DataFrame(events)
    last_event_date_str = events[-1]['Date'] if events else None
    if last_event_date_str:
        last_event_date = datetime.strptime(last_event_date_str, "%a%b %d")
        last_event_date = last_event_date.replace(year=datetime.now().year)
        if datetime.now() > last_event_date:
            next_start_date = last_event_date + timedelta(days=1)
            return None, next_start_date

    return df, None

# Generate a calendar table for Streamlit
def create_calendar_table(events_df, year, month):
    cal = pd.DataFrame(columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    month_days = calendar.monthcalendar(year, month)

    for week in month_days:
        week_events = []
        for day in week:
            if day == 0:
                week_events.append("")
            else:
                date_str = f"{year}-{month:02d}-{day:02d}"
                if date_str in events_df.index:
                    events = events_df.loc[date_str]
                    week_events.append(events)
                else:
                    week_events.append(day)
        cal = cal.append(pd.Series(week_events, index=cal.columns), ignore_index=True)

    return cal

# Streamlit page function
def market_calendar_page():
    st.title("Market Calendar")

    # Fetch earnings data
    fetcher = EventsFetcher()
    df_earnings = fetcher.get_next_earnings(limit=5)

    # Scrape Forex Factory calendar
    base_url = "https://www.forexfactory.com/calendar"
    url = base_url
    while True:
        df_forex, next_start_date = scrape_calendar(url)
        if df_forex is not None:
            break
        if next_start_date:
            url = f"{base_url}?week={next_start_date.strftime('%b%d.%Y').lower()}"

    if df_forex is None:
        st.warning("No Forex data found.")
        return

    if df_earnings.empty:
        st.warning("No earnings data found.")
        return

    # Filter data
    high_impact_events = df_forex[df_forex['Impact'] == 'High Impact']
    large_cap_earnings = df_earnings[df_earnings['marketcap'] > 50e9]

    # Convert date columns to datetime
    high_impact_events['Date'] = pd.to_datetime(high_impact_events['Date'])
    large_cap_earnings['Date'] = pd.to_datetime(large_cap_earnings['Date'])

    # Combine both DataFrames
    combined_df = pd.concat([high_impact_events, large_cap_earnings])

    # Create a pivot table with the dates as index
    pivot_table = combined_df.pivot_table(
        index='Date', 
        values='Event',  # or any other column you'd like to display
        aggfunc=lambda x: ' | '.join(x)
    )

    # Generate a calendar table
    year = datetime.now().year
    month = datetime.now().month
    calendar_table = create_calendar_table(pivot_table, year, month)

    # Display the table in Streamlit
    st.write(f"Economic Events and Earnings Calendar - {calendar.month_name[month]} {year}")
    st.table(calendar_table)

# Call the function to render the page
if __name__ == "__main__":
    market_calendar_page()

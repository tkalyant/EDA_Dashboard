import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import time
from datetime import datetime

# Initialize the session state for the warning message
if 'show_warning' not in st.session_state:
    st.session_state['show_warning'] = True
    st.session_state['warning_start_time'] = time.time()

# Function to show a warning message with a dismiss button
def show_timed_warning(message, duration=10):
    if st.session_state['show_warning']:
        elapsed_time = time.time() - st.session_state['warning_start_time']
        if elapsed_time <= duration:
            st.warning(message)
        else:
            st.session_state['show_warning'] = False


# Database connection function
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            passwd="workspace4321",
            database="stockapp"
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

# Function to check if data already exists
def check_data_exists(connection, symbol, date):
    cursor = connection.cursor()
    query = "SELECT COUNT(*) FROM stocks WHERE symbol = %s AND date = %s"
    cursor.execute(query, (symbol, date))
    result = cursor.fetchone()
    cursor.close()
    return result[0] > 0

# Function to execute a query
def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        connection.commit()
    except Error as e:
        print(f"The error '{e}' occurred")
        return False
    finally:
        cursor.close()
    return True

# Function to fetch stock data from the database
def get_stock_data_from_db(connection, symbol, start_date, end_date):
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM stocks WHERE symbol = %s AND date >= %s AND date <= %s"
        cursor.execute(query, (symbol, start_date, end_date))
        result = cursor.fetchall()
        return pd.DataFrame(result)
    except Error as e:
        print(f"Error fetching data from database: {e}")
        return pd.DataFrame()
    
# Function to insert stock data into the database
def insert_stock_data(connection, symbol, stock_data):
    for index, row in stock_data.iterrows():
        if not check_data_exists(connection, symbol, index.date()):
            query = """INSERT INTO stocks (symbol, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            data = (symbol, index.date(), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
            execute_query(connection, query, data)

# Function to download stock data from Yahoo Finance and store in the database
def download_and_store_stock_data(connection, symbol, period):
    data = yf.download(symbol, period=period)
    data.reset_index(inplace=True)
    data['Symbol'] = symbol

    cursor = connection.cursor()
    for index, row in data.iterrows():
        query = "INSERT INTO stocks (symbol, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (row['Symbol'], row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

    connection.commit()

# Main data analysis function
def data_analysis():
    if 'warning_active' not in st.session_state:
        st.session_state.warning_active = False

    if st.session_state.warning_active and datetime.now() > st.session_state.warning_end_time:
        st.experimental_rerun()

    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database.")
        return

    st.sidebar.header("Stock Data Analysis")
    symbol = st.sidebar.text_input("Enter Stock Symbol", value='AAPL')

    start_date = st.sidebar.date_input("Start Date", value=datetime.now().date())
    end_date = st.sidebar.date_input("End Date", value=datetime.now().date())

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    data = get_stock_data_from_db(connection, symbol, start_date_str, end_date_str)
    if data.empty:
        show_timed_warning("No Data Available for the selected symbol and period. However, you can get that from Yahoo Finance.")
        if st.sidebar.button("Fetch and Store Data"):
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            if not stock_data.empty:
                insert_stock_data(connection, symbol, stock_data)
                st.success("Data fetched and stored successfully.")
                data = get_stock_data_from_db(connection, symbol, start_date_str, end_date_str)
            else:
                st.error("No data available from Yahoo Finance for the given symbol and period.")

    if not data.empty:
        st.header(f'Visualization for {symbol}')
    
        # Ensure the 'date' column is in datetime format
        data['date'] = pd.to_datetime(data['date'])

        fig = go.Figure(data=[go.Candlestick(x=data['date'],
                                         open=data['open'],
                                         high=data['high'],
                                         low=data['low'],
                                         close=data['close'],
                                         name='Market Data')])
        fig.update_layout(title='Stock Price Evolution', yaxis_title='Stock Price (USD per share)')
        st.plotly_chart(fig)
if __name__ == "__main__":
    data_analysis()
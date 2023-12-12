import urllib
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from data_analysis import create_connection, get_stock_data_from_db
import datetime

def main():
    connection = create_connection()
    st.sidebar.title("Stock Market Data Analysis")
    symbol = st.sidebar.text_input("Enter Stock Symbol", value='AAPL')
    start_date = st.sidebar.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

    box = st.selectbox("Information", ["Stock Market Info", "Covid-19 impact"])
    if box == "Stock Market Info":
        readme_text = st.markdown(get_file_content_as_string("README.md"))
    elif box == "Covid-19 impact":
        st.header(f"Impact of COVID-19 on {symbol} Stock Market")
        df1 = get_stock_data_from_db(connection, symbol, start_date, end_date)
        if not df1.empty:
            fig = go.Figure(data=[go.Candlestick(x=df1['date'],
                                                 open=df1['open'],
                                                 high=df1['high'],
                                                 low=df1['low'],
                                                 close=df1['close'],
                                                 name='Market Data')])
            fig.update_layout(title=f'{symbol} Stock Volatility', yaxis_title='Stock Price', width=800, height=550)
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig)
        else:
            st.write("No data available for the selected symbol and period.")
        readme_text_1 = st.markdown(get_file_content_as_string("Covid.md"))

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Lakshya-Ag/Streamlit-Dashboard/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if __name__ == "__main__":
    main()

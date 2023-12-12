import urllib
import xlrd
import streamlit as st
import mysql.connector
import smtplib
import random
import string
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import seaborn as sns
import plotly.graph_objs as go
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import matplotlib.pyplot as plt
import mysql.connector
from data_analysis import create_connection, get_stock_data_from_db, show_timed_warning,insert_stock_data
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import time
from datetime import datetime
import urllib
import xlrd
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import seaborn as sns
import plotly.graph_objs as go
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
plt.style.use('bmh')
import quandl
import matplotlib.animation as ani
import altair as alt
import urllib
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from data_analysis import create_connection, get_stock_data_from_db
import datetime


########################################################################################################################
# Database connection
def connect_to_db():
    conn = mysql.connector.connect(
       host="127.0.0.1",
            user="root",
            passwd="workspace4321",
            database="stockapp"
    )
    return conn

# Create user table
def create_usertable():
    conn = connect_to_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(255),
            password VARCHAR(255)
        )
    ''')
    conn.commit()
    conn.close()
    
def add_userdata(username, password):
    conn = connect_to_db()
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, password))
    conn.commit()
    conn.close()
    
# Login user
def login_user(username, password):
    conn = connect_to_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
    data = c.fetchall()
    conn.close()
    return data

# View all users
def view_all_users():
    conn = connect_to_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    data = c.fetchall()
    conn.close()
    return data

def add_userdata(username, password, email, verification_code, verified=False):
    conn = connect_to_db()  # Function to connect to your MySQL database
    c = conn.cursor()

    # SQL query to insert new user data
    query = "INSERT INTO users (username, password, email, verification_code, verified) VALUES (%s, %s, %s, %s, %s)"
    c.execute(query, (username, password, email, verification_code, verified))

    conn.commit()
    conn.close()
    
def add_user_to_database(username, password, email=None):
    conn = connect_to_db()
    c = conn.cursor()

    # Assuming your users table has columns for username, password, and optionally email
    query = "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)"
    c.execute(query, (username, password, email))

    conn.commit()
    conn.close()


def remove_all_user(username, password):
    conn = connect_to_db()
    c = conn.cursor()
    c.execute('DELETE FROM users');
    conn.commit()

###################################################################################################
# Function to send email
def send_verification_email(user_email, verification_code):
    sender_email = "eda878327@gmail.com"  # Replace with your email
    sender_password = "eda121223"  # Replace with your email password

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    subject = "Verify Your Email"
    body = f"Your verification code is: {verification_code}"

    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(sender_email, user_email, message)
    server.quit()

# Function to generate a random verification code
def generate_verification_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def signup_user():
    new_user = st.text_input("Username")
    new_email = st.text_input("Email")  # Optional, depending on whether you still want to collect emails
    new_password = st.text_input("Password", type='password')
    new_con_password = st.text_input("Confirm Password", type='password')

    if st.button("Signup"):
        if new_password == new_con_password:
            # Add user data to the database (assuming a function like 'add_user_to_database' exists)
            add_user_to_database(new_user, new_password, new_email)
            st.success("You have successfully created an account. You can now log in.")
        else:
            st.warning("Password and Confirm Password don't match.")

def verify_user():
    user_input_code = st.text_input("Enter verification code")
    if st.button("Verify Code"):
        if user_input_code == st.session_state['verification_code']:
            # Activate the user in the database
            # ...
            st.success("Email verified! Your account is now active.")
            del st.session_state['verify_email']  # Remove the verification flag
            del st.session_state['verification_code']  # Remove the stored code
            # Redirect to login or main page
        else:
            st.error("Incorrect verification code. Please try again.")

            
def check_user_exists(username):
    conn = connect_to_db()  # Function to connect to your MySQL database
    c = conn.cursor(buffered=True)
    
    # SQL query to check if the user exists
    query = "SELECT * FROM users WHERE username = %s"
    c.execute(query, (username,))
    
    result = c.fetchone()
    conn.close()

    return result is not None


####################################################################################################
   
def main():
    st.title("Stock prediction EDA Dashboard with User authentication")
    # st.title("Login App")
    menu = ["Home", "Login", "Signup", "Admin"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        readme_text = st.markdown(get_file_content_as_string("README.md"))
        # st.subheader("Home")
        
    elif choice == "Login":
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            create_usertable()
        
            # Check if the user exists in the database
            user_exists = check_user_exists(username)

            if user_exists:
                # Verify login credentials
                result = login_user(username, password)
                if result:
                    st.success("Logged In As {}".format(username))
                    # Proceed with the main functionality
                    mainfunc()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("User does not exist. Please sign up.")
            
    elif choice == "Signup":
        signup_user()


#########################################################################################################
def mainfunc():

    st.sidebar.header("What To Do")
    app_mode = st.selectbox("Select the app mode", ["Home", "Data Analysis", "Prediction", "Show the Code"])

    if app_mode == "Home":
        st.success("Select Data Analysis or prediction to move on")
        readme_text = st.markdown(get_file_content_as_string("README.md"))
    elif app_mode == "Data Analysis":
        # readme_text.empty()
        data_analysis()
    elif app_mode == "Prediction":
        # readme_text.empty()
        prediction()
    elif app_mode == "Show the Code":
        # readme_text.empty()
        st.code(get_file_content_as_string("myapp.py"))

#####################################################################################################################

companies = {}
xls = xlrd.open_workbook("cname.xls")
sh = xls.sheet_by_index(0)
for i in range(505):
    cell_value_class = sh.cell(i, 0).value
    cell_value_id = sh.cell(i, 1).value
    companies[cell_value_class] = cell_value_id

############################################################################

def company_name():
    company = st.sidebar.selectbox("Companies", list(companies.keys()), 0)
    return company
# company = company_name()

############################################################################

def show_data():
    show = st.sidebar.selectbox("Options", ["Graphs", "Company Data"], 0)
    return show
# show_data = show_data()

############################################################################

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Lakshya-Ag/Streamlit-Dashboard/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

############################################################################
@st.cache(suppress_st_warning=True)
def prediction_graph(algo, confidence, cdata):
    st.header(algo + ', Confidence score is ' + str(round(confidence, 2)))
    fig6 = go.Figure(data=[go.Scatter(x=list(cdata.index), y=list(cdata.Close), name='Close'),
                           # go.Scatter(x=list(chart_data.index), y=list(chart_data.Vclose), name='Vclose'),
                           go.Scatter(x=list(cdata.index), y=list(cdata.Vpredictions),
                                      name='Predictions')])

    fig6.update_layout(width=850, height=550)
    fig6.update_xaxes(rangeslider_visible=True,
                      rangeselector=dict(
                          buttons=list([
                              dict(count=30, label="30D", step="day", stepmode="backward"),
                              dict(count=60, label="60D", step="day", stepmode="backward"),
                              dict(count=90, label="90D", step="day", stepmode="backward"),
                              dict(count=120, label="120D", step="day", stepmode="backward"),
                              dict(count=150, label="150D", step="day", stepmode="backward"),
                              dict(step="all")
                          ])
                      ))
    st.plotly_chart(fig6)

#############################################################################

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

######################################################################################


def prediction():
    def data_download():
        company = company_name()
        data = yf.download(tickers=companies[company], period='200d', interval='1d')
        #print(type(data.index))


        def divide(j):
            j = j / 1000000
            return j

        data['Volume'] = data['Volume'].apply(divide)
        data.rename(columns={'Volume': 'Volume (in millions)'}, inplace=True)
        return data

    df = data_download()

    pred = st.sidebar.radio("Regression Type", ["Tree Prediction", "Linear Regression", "SVR Prediction",
                                                "RBF Prediction", "Polynomial Prediction", "Linear Regression 2"])

    # removing index which is date
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # rearranging the columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume (in millions)']]
    df['Close'] = scaler.fit_transform(df[['Close']])
    df = df[['Close']]

    # create a variable to predict 'x' days out into the future
    future_days = 50
    # create a new column( target) shifted 'x' units/days up
    df['Prediction'] = df[['Close']].shift(-future_days)

    # create the feature data set (x) and convet it to a numpy array and remove the last 'x' rows
    x = np.array(df.drop(['Prediction'], axis =1))[:-future_days]

    # create a new target dataset (y) and convert it to a numpy array and get all of the target values except the last'x' rows)
    y = np.array(df['Prediction'])[:-future_days]

    # split the data into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # create the models
    # create the decision treee regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    # create the linear regression model
    lr = LinearRegression().fit(x_train, y_train)

    # create the svr model
    svr_rbf = SVR(C=1e3, gamma=.1)
    svr_rbf.fit(x_train, y_train)

    # create the RBF model
    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=.85)
    rbf_svr.fit(x_train, y_train)

    # Create the polyomial model
    poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
    poly_svr.fit(x_train, y_train)

    # create the linear 2 model
    lin_svr = SVR(kernel='linear', C=1000.0, gamma=.85)
    lin_svr.fit(x_train, y_train)

    # get the last x rows of the feature dataset
    x_future = df.drop(['Prediction'], axis= 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    # show the model tree prediction
    tree_prediction = tree.predict(x_future)

    # show the model linear regression prediction
    lr_prediction = lr.predict(x_future)

    # show the model SVR prediction
    SVR_prediction = svr_rbf.predict(x_future)

    # show the model RBF prediction
    RBF_prediction = rbf_svr.predict(x_future)

    # show the model Polynomial prediction
    poly_prediction = poly_svr.predict(x_future)

    ##show thw model linear regression2 prediction
    lr2_prediction = lin_svr.predict(x_future)

    if pred == "Linear Regression":
        predictions = lr_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close
        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]
        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        lin_confidence = lr.score(x_test, y_test)
        prediction_graph(pred, lin_confidence, chart_data)

    elif pred == "Tree Prediction":
        predictions = tree_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close

        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]

        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        tree_confidence = tree.score(x_test, y_test)
        prediction_graph(pred, tree_confidence, chart_data)

    elif pred == "SVR Prediction":
        predictions = SVR_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close

        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]

        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        svr_confidence = svr_rbf.score(x_test, y_test)
        prediction_graph(pred, svr_confidence, chart_data)

    elif pred == "RBF Prediction":
        predictions = RBF_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close

        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]

        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        rbf_confidence = rbf_svr.score(x_test, y_test)
        prediction_graph(pred, rbf_confidence, chart_data)

    elif pred == "Polynomial Prediction":
        predictions = poly_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close

        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]

        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        poly_confidence = poly_svr.score(x_test, y_test)

    elif pred == "Linear Regression 2":
        predictions = lr2_prediction
        valid = df[x.shape[0]:]
        valid['predictions'] = predictions

        # alter
        data = {'Close': [], 'Vclose': [], 'Vpredictions': []}
        mod = pd.DataFrame(data)
        mod.set_index = 'index'
        mod.Close = df.Close

        # mod.Vclose = df.Close.loc[:747]
        # mod.Vpredictions = df.Close.loc[:747]

        # mod.Vclose.loc[148:] = valid.Close
        mod.Vpredictions.loc[148:] = valid.predictions
        # mod.Close = df.Close.loc[:150]
        chart_data = mod
        linsvr_confidence = lin_svr.score(x_test, y_test)
        prediction_graph(pred, linsvr_confidence, chart_data)

##################################################################################

if __name__ == "__main__":
    main()

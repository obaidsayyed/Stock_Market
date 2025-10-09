import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import date, timedelta
from math import sqrt
import plotly.graph_objects as go

# Set background image 
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://photos.peopleimages.com/picture/202309/2917845-abstract-financial-graph-with-candlestick-chart-in-stock-market-on-dark-background-fit_400_400.jpg");
        background-size: cover;
        background-position: center;
    }

    /* Make all text white */
    * {
        color: white !important;
    }

    /* Optional: make headings slightly bolder */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700;
    }

    /* Optional: make Streamlit text elements white */
    .stMarkdown, .stText, .stMetric, .stDataFrame, .stPlotlyChart {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(page_title="Intra-Day Predictor by Obaid Sayyed", layout="wide")
st.title("📈 Stock Price Predictor")
st.markdown(
    "<small>⚠️ Note: The model by default shows results for Google (Class A), please ennter your stock.</small>", 
    unsafe_allow_html=True
)

#Sidebar
st.sidebar.header("Settings")

# Stock Symbol input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol Only (e.g., GOOG, TATAMOTORS.NS, AAPL), Google to find for your Stock's Ticker Name","GOOGL")

# Time range selector
time_option = st.sidebar.selectbox(
    "Select Time Range for Chart",
    ("Last 6 Months", "Last 1 Year", "Last 5 Years", "All Data")
)

# Determine start date based on selection
today = date.today()
if time_option == "Last 6 Months":
    start_date = today - timedelta(days=182)
elif time_option == "Last 6 Months":
    start_date = today - timedelta(days=182)
elif time_option == "Last 1 Year":
    start_date = today - timedelta(days=365)
elif time_option == "Last 5 Years":
    start_date = today - timedelta(days=1825)
else:
    start_date = date(2015, 1, 1)  

#Fetch Data
data = yf.download(stock_symbol, start=start_date, end=today, auto_adjust=True)
data.columns = data.columns.droplevel(1)
data = data.dropna()

#Feature Engineering
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)
data['5_day_avg'] = data['Close'].rolling(window=5).mean()
data['10_day_avg'] = data['Close'].rolling(window=10).mean()

def calculate_RSI(price, period=14):
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_RSI(data['Close'])
ema_short = data['Close'].ewm(span=12, adjust=False).mean()
ema_long = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_short - ema_long
data['MA20'] = data['Close'].rolling(window=20).mean()
data['BB_upper'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
data['BB_lower'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()
data = data.dropna()

#Features and Target
X = data[['Lag1','Lag2','Lag3','5_day_avg','10_day_avg','RSI','MA20','BB_lower','BB_upper','MACD']]
y = data['Close']

#Train Model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

#Model Evaluation
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y, y_pred)

#Next Day Prediction
last_row = data.iloc[-1]
next_features = np.array([last_row['Lag1'], last_row['Lag2'], last_row['Lag3'],
                          last_row['5_day_avg'], last_row['10_day_avg'],
                          last_row['RSI'], last_row['MA20'], last_row['BB_lower'],
                          last_row['BB_upper'], last_row['MACD']]).reshape(1, -1)
next_day_pred = model.predict(next_features)[0]

#High-Low Prediction
latest_std = data['Close'].rolling(window=20).std().iloc[-1]
predicted_high = next_day_pred + latest_std
predicted_low = next_day_pred - latest_std

#Display Prediction
st.subheader(f"Predicted Next Day Close Price for {stock_symbol}:")
st.write(f"💰 Predicted Close: **{next_day_pred:.3f}**")
st.write(f"📈 Predicted High: **{predicted_high:.3f}**")
st.write(f"📉 Predicted Low: **{predicted_low:.3f}**")


#Display Model Metrics
st.subheader("Model Evaluation Metrics")
st.write(f"Average Deviation by Model: {rmse:.3f}")
st.write(f"Accuracy: {r2*100:.3f}")

#Plotly Charts

# Historical Closing Prices
st.subheader("Historical Closing Prices")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Closing Price'
))
fig1.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis=dict(rangeslider=dict(visible=True)),
    yaxis=dict(fixedrange=False),
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# Actual vs Predicted Prices
st.subheader("Actual vs Predicted Prices")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=data.index,
    y=y,
    mode='lines',
    name='Actual Price',
    line=dict(color='blue')
))
fig2.add_trace(go.Scatter(
    x=data.index,
    y=y_pred,
    mode='lines',
    name='Predicted Price',
    line=dict(color='red')
))
fig2.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis=dict(rangeslider=dict(visible=True)),
    yaxis=dict(fixedrange=False),
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

#python -m streamlit run app.py



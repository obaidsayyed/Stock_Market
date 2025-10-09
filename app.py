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
        background-image: url("data:image/jpeg;base64,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");
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



import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import date 
from math import sqrt

today=date.today()

name = input("Enter the Stock Symbol for the stock you are intrested in: ")
data=yf.download(name,start='2015-01-01',end= today,auto_adjust=True)

print(f"Showing data for {name}")
data.columns = data.columns.droplevel(1)


plt.figure(figsize=(10,6))
plt.plot(data['Close'], label=f'{name} Closing Price')
plt.title(f'{name} Stock Price (2015–Present)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)
data['5_day_avg'] = data['Close'].rolling(window=5).mean()
data['10_day_avg'] = data['Close'].rolling(window=10).mean()

def calculate_RSI(price,period=14):
    delta=price.diff()
    gain=delta.clip(lower=0)
    loss= -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_RSI(data['Close'], period=14)


ema_short = data['Close'].ewm(span=12, adjust=False).mean()
ema_long = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_short - ema_long


data['MA20'] = data['Close'].rolling(window=20).mean()
data['BB_upper'] = data['MA20'] + 2*data['Close'].rolling(window=20).std()
data['BB_lower'] = data['MA20'] - 2*data['Close'].rolling(window=20).std()

data = data.dropna()


#short term patterns
X = data[['Lag1','Lag2','Lag3', '5_day_avg', '10_day_avg','RSI','MA20','BB_lower','BB_upper','MACD']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#model training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation for {name}:")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {sqrt(mse):.3f}")
print(f"R² Score: {r2:.3f}")


#Visualize Actual vs Predicted
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='black')
plt.title(f'Actual vs Predicted Price for {name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


#next day prediction
last_row = data.iloc[-1]
next_features = np.array([last_row['Lag1'],last_row['Lag2'],last_row['Lag3'], 
                          last_row['5_day_avg'], last_row['10_day_avg'],
                          last_row['RSI'],last_row['MA20'],last_row['BB_lower'],
                          last_row['BB_upper'],last_row['MACD']]).reshape(1, -1)
next_day_prediction = model.predict(next_features)[0]
print(f"\nPredicted Next Day Close Price for {name}: {next_day_prediction:.3f}")

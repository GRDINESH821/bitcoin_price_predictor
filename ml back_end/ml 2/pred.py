import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime, timedelta
import joblib

# Define the date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

# Download data for multiple tickers
tickers = ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F', 'USDEUR=X', 'USDCAD=X', 'CL=F', '^IXIC']
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Extract relevant columns (Open, Close, High, Low) for each asset and rename columns
bitcoin_data = data['BTC-USD'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Bitcoin_{x}')
ethereum_data = data['ETH-USD'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Ethereum_{x}')
sp500_data = data['^GSPC'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'SP500_{x}')
gold_data = data['GC=F'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Gold_{x}')
usd_eur_data = data['USDEUR=X'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'USDEUR_{x}')
usd_cad_data = data['USDCAD=X'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'USDCAD_{x}')
oil_data = data['CL=F'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Oil_{x}')
nasdaq_data = data['^IXIC'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'NASDAQ_{x}')

combined_data = pd.concat([bitcoin_data, ethereum_data, sp500_data, gold_data, usd_eur_data, usd_cad_data, oil_data, nasdaq_data], axis=1)

# Fill missing values using forward fill method
combined_data_filled = combined_data.fillna(method='ffill').fillna(method='bfill')

# Feature Engineering
data = combined_data_filled
data['SMA_10'] = data['Bitcoin_Close'].rolling(window=10).mean()
data['EMA_9'] = data['Bitcoin_Close'].ewm(span=9, adjust=False).mean()
data['Volatility'] = data['Bitcoin_High'] - data['Bitcoin_Low']

for i in range(1, 8):
    data[f'High_Day_{i}'] = data['Bitcoin_High'].shift(-i)
    data[f'Low_Day_{i}'] = data['Bitcoin_Low'].shift(-i)
    data[f'Close_Day_{i}'] = data['Bitcoin_Close'].shift(-i)

# Add day number relative to the start of the dataset as a feature
first_date = data.index.min()
data['DayNumber'] = (data.index - first_date).days

# Split features (X) and target variables (y)
X = data.drop([f'High_Day_{i}' for i in range(1, 8)] + [f'Low_Day_{i}' for i in range(1, 8)] + [f'Close_Day_{i}' for i in range(1, 8)], axis=1)
y = data[[f'High_Day_{i}' for i in range(1, 8)] + [f'Low_Day_{i}' for i in range(1, 8)] + [f'Close_Day_{i}' for i in range(1, 8)]]

X.interpolate(method='linear', inplace=True)
y.interpolate(method='linear', inplace=True)

X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split data into training and testing sets.
X_train, X_test = X.iloc[:-7], X.iloc[-7:]
y_train, y_test = y.iloc[:-7], y.iloc[-7:]

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_mlp_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(21))  # 21 output neurons for 3 targets x 7 days
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create and train the MLP model
model = KerasRegressor(build_fn=create_mlp_model, verbose=4)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save the model
model.model_.save('bitcoin_model.h5')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
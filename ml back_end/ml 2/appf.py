from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('bitcoin_model.h5')
scaler = joblib.load('scaler.pkl')

# Load the dataset and preprocess it (similar to the training script)
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)
tickers = ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F', 'USDEUR=X', 'USDCAD=X', 'CL=F', '^IXIC']
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

bitcoin_data = data['BTC-USD'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Bitcoin_{x}')
ethereum_data = data['ETH-USD'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Ethereum_{x}')
sp500_data = data['^GSPC'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'SP500_{x}')
gold_data = data['GC=F'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Gold_{x}')
usd_eur_data = data['USDEUR=X'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'USDEUR_{x}')
usd_cad_data = data['USDCAD=X'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'USDCAD_{x}')
oil_data = data['CL=F'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'Oil_{x}')
nasdaq_data = data['^IXIC'][['Open', 'Close', 'High', 'Low']].rename(columns=lambda x: f'NASDAQ_{x}')

combined_data = pd.concat([bitcoin_data, ethereum_data, sp500_data, gold_data, usd_eur_data, usd_cad_data, oil_data, nasdaq_data], axis=1)
combined_data_filled = combined_data.fillna(method='ffill').fillna(method='bfill')

data = combined_data_filled
data['SMA_10'] = data['Bitcoin_Close'].rolling(window=10).mean()
data['EMA_9'] = data['Bitcoin_Close'].ewm(span=9, adjust=False).mean()
data['Volatility'] = data['Bitcoin_High'] - data['Bitcoin_Low']

for i in range(1, 8):
    data[f'High_Day_{i}'] = data['Bitcoin_High'].shift(-i)
    data[f'Low_Day_{i}'] = data['Bitcoin_Low'].shift(-i)
    data[f'Close_Day_{i}'] = data['Bitcoin_Close'].shift(-i)

first_date = data.index.min()
data['DayNumber'] = (data.index - first_date).days

X = data.drop([f'High_Day_{i}' for i in range(1, 8)] + [f'Low_Day_{i}' for i in range(1, 8)] + [f'Close_Day_{i}' for i in range(1, 8)], axis=1)

def get_bitcoin_opening_price(input_date):
    input_dateC = pd.to_datetime(input_date).date()
    end_date = input_dateC + timedelta(days=1)
    start_date = input_dateC - timedelta(days=1)
    data = yf.download('BTC-USD', start=start_date, end=end_date)
    if input_date in data.index:
        opening_price = data.loc[input_date, 'Open']
        return opening_price
    else:
        return None

def swing_trading_strategy_max_profit(open_price, high_prices, low_prices, closing_price_7th_day):
    initial_bitcoins = 100000.00 / open_price
    max_final_value = initial_bitcoins * closing_price_7th_day  # if no trades are made
    best_sell_day = None
    best_buy_day = None
    # coins_present = initial_bitcoins
    # Iterate over each day as a potential sell day
    for i in range(7):
        # Calculate cash if sold on day i at the high price
        sell_cash = initial_bitcoins * high_prices[i]
        # Check if holding the cash until the end of the 7th day is better
        if sell_cash > max_final_value:
            max_final_value = sell_cash
            best_sell_day = i
            best_buy_day = None

        # Iterate over each subsequent day as a potential buy day
        for j in range(i + 1, 7):
            # Calculate bitcoins if bought on day j at the low price
            buy_bitcoins = sell_cash / low_prices[j]
            # Calculate the final value in bitcoins on the 7th day
            final_value = buy_bitcoins * closing_price_7th_day
            # Update the best strategy if this combination yields a higher final value
            if final_value > max_final_value:
                max_final_value = final_value
                best_sell_day = i
                best_buy_day = j

    return {
        'sell_day': best_sell_day,
        'buy_day': best_buy_day,
        'final_value': max_final_value
    }

@app.route('/predict', methods=['GET'])
def predict():
    input_date_str = request.args.get('date')
    if not input_date_str:
        return jsonify({"error": "Please provide a date in the format yyyy-mm-dd"}), 400

    try:
        input_date = pd.Timestamp(input_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use yyyy-mm-dd"}), 400

    if not (data.index.min() <= input_date <= data.index.max()):
        return jsonify({"error": "Date out of valid range for predictions"}), 400

    input_day_number = (input_date - first_date).days
    input_row = X[X['DayNumber'] == input_day_number].iloc[0]
    X_input = pd.DataFrame([input_row])
    X_input = X_input.reindex(columns=X.columns, fill_value=0)
    X_input = scaler.transform(X_input)
    predictions = model.predict(X_input)
    
    highest_price = float(np.max(predictions[0][0:7]))
    lowest_price = float(np.min(predictions[0][7:14]))
    avg_closing_price = float(np.mean(predictions[0][14:21]))

    return jsonify({
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "average_closing_price": avg_closing_price
    })

@app.route('/trade', methods=['GET'])
def trade():
    input_date_str = request.args.get('date')
    if not input_date_str:
        return jsonify({"error": "Please provide a date in the format yyyy-mm-dd"}), 400

    try:
        input_date = pd.Timestamp(input_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use yyyy-mm-dd"}), 400

    opening_price = get_bitcoin_opening_price(input_date_str)
    if opening_price is None:
        return jsonify({"error": "Could not retrieve opening price for the given date"}), 400

    input_day_number = (input_date - first_date).days
    input_row = X[X['DayNumber'] == input_day_number].iloc[0]
    X_input = pd.DataFrame([input_row])
    X_input = X_input.reindex(columns=X.columns, fill_value=0)
    X_input = scaler.transform(X_input)
    predictions = model.predict(X_input)
    high_prices = predictions[0][0:7].astype(float)
    low_prices = predictions[0][7:14].astype(float)
    closing_price_7th_day = float(predictions[0][20])  # Correct index for closing price on 7th day

    strategy = swing_trading_strategy_max_profit(opening_price, high_prices, low_prices, closing_price_7th_day)
    return jsonify(strategy)

if __name__ == '__main__':
    app.run(debug=True)
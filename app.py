from flask import Flask, render_template
import numpy as np
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features
import matplotlib.pyplot as plt
import ta

app = Flask(__name__)

def get_current_btc_price():
  try:
      # API endpoint for BTC price
      url = "https://api.coindesk.com/v1/bpi/currentprice.json"

      # Send a GET request to the API
      response = requests.get(url)
      data = response.json()

      # Extract the current BTC price
      btc_price = data["bpi"]["USD"]["rate"]

      return float(btc_price.replace(",", ""))
  except Exception as e:
      return f"Error fetching BTC price: {str(e)}"



api_keys = ["6MPAZEBBFVVFSQ60", "9YMRLVKLJ1E0SZGT", "IF9FE6Q570V8UOBH"] 
def get_alpha_vantage_btc_history(api_keys):
    for api_key in api_keys:  # Iterate through the list of keys
        try:
            ts = TimeSeries(key=api_key)
            btc_data, meta_data = ts.get_daily(symbol='BTCUSD', outputsize='full')
            # Extract timestamps and all relevant data points
            timestamps = list(btc_data.keys())
            open_prices = [float(btc_data[ts]['1. open']) for ts in timestamps]
            high_prices = [float(btc_data[ts]['2. high']) for ts in timestamps]
            low_prices = [float(btc_data[ts]['3. low']) for ts in timestamps]
            close_prices = [float(btc_data[ts]['4. close']) for ts in timestamps]
            volumes = [int(btc_data[ts]['5. volume']) for ts in timestamps]
            # Create a DataFrame
            df = pd.DataFrame({
                "Timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
                "close": close_prices
            })
            df["Date"] = pd.to_datetime(df["Timestamp"])
            # Save to CSV
            df.to_csv("btc_price_data_alpha_vantage.csv", index=False)
            dfr = df[::-1]
            dfr.to_csv("btc_price_data_alpha_vantage_ful.csv", index=False)
            print("Saved full BTC price data from Alpha Vantage to btc_price_data_alpha_vantage_full.csv")
            return dfr
        except Exception as e:
            print(f"Error fetching BTC price using {api_key}: {str(e)}")
            # Move on to the next API key in the list

    # If all keys fail, raise an error
    raise Exception("Failed to fetch BTC data with all provided API keys.") 

btc_history = get_alpha_vantage_btc_history(api_keys)



# Load historical BTC price data
btc_data = pd.read_csv("btc_price_data_alpha_vantage_ful.csv")

def predict_price_trend(btc_data, period=5):
    # Calculate moving averages
    btc_data["SMA_20"] = btc_data["close"].rolling(window=20).mean()
    btc_data["EMA_50"] = btc_data["close"].ewm(span=50, adjust=False).mean()

    # Calculate RSI
    btc_data = add_all_ta_features(btc_data, "open", "high", "low", "close", "volume", fillna=True)
    btc_data["RSI"] = btc_data["momentum_rsi"]

    # Calculate MACD
    btc_data["EMA_12"] = btc_data["close"].ewm(span=12, adjust=False).mean()
    btc_data["EMA_26"] = btc_data["close"].ewm(span=26, adjust=False).mean()
    btc_data["MACD"] = btc_data["EMA_12"] - btc_data["EMA_26"]
    btc_data["Signal_Line"] = btc_data["MACD"].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    btc_data["Upper_Band"], btc_data["Lower_Band"] = (
        btc_data["SMA_20"] + 2 * btc_data["close"].rolling(window=20).std(),
        btc_data["SMA_20"] - 2 * btc_data["close"].rolling(window=20).std(),
    )

    # Calculate ADX
    btc_data["ADX"] = ta.trend.ADXIndicator(
        btc_data["high"], btc_data["low"], btc_data["close"], window=14
    ).adx()

    # Calculate Stochastic Oscillator
    btc_data["Stochastic_K"] = (
        (btc_data["close"] - btc_data["low"].rolling(window=14).min())
        / (btc_data["high"].rolling(window=14).max() - btc_data["low"].rolling(window=14).min())
    ) * 100

    # Prepare features for prediction
    X = btc_data[["SMA_20", "EMA_50", "RSI", "MACD", "ADX", "Stochastic_K"]]
    y = btc_data["close"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    X = imputer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    model = RandomForestRegressor(n_estimators=270, max_depth=14)
    model.fit(X_train, y_train)

    # Predict the next BTC price
    next_price = model.predict([[btc_data["SMA_20"].iloc[-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                 btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])

    if period == 5:
        # Predict prices for the next 5 days
        five_day_prices = [next_price[0]]
        for i in range(1, period):
            next_price = model.predict([[five_day_prices[i-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                         btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])
            five_day_prices.append(next_price[0])

        return five_day_prices

    return next_price[0]

@app.route('/')
def index():
    # Fetch current price
    current_price = get_current_btc_price()

    # Predict prices
    tomorrow_price = predict_price_trend(btc_data)
    five_day_prices = predict_price_trend(btc_data, period=5)

    # Convert NumPy arrays to floats
    tomorrow_price = float(tomorrow_price[0])
    five_day_prices = [float(price) for price in five_day_prices]

    # Process the list of five-day prices 
    five_day_prices_with_index = enumerate(five_day_prices)

    # Calculate price comparison and recommendations
    price_comparison = ""
    recommendation = ""
    if tomorrow_price > current_price:
        percentage_increase = round(((tomorrow_price - current_price) / current_price) * 100, 2)
        price_comparison = f"Tomorrow's price is predicted to be {percentage_increase}% higher than today's price."
        if percentage_increase > 0.2:
            recommendation = "Buy 10% of your BTC amount."
        else:
            recommendation = "Buy a small percentage of your current BTC like 4 to 2 percent, or nothing."

    elif tomorrow_price < current_price:
        percentage_decrease = round(((current_price - tomorrow_price) / current_price) * 100, 2)
        price_comparison = f"Tomorrow's price is predicted to be {percentage_decrease}% lower than today's price."
        if percentage_decrease > 0.1:
            recommendation = "Sell 5% of your BTC."
        else:
            recommendation = "Do nothing or sell a really small percentage of BTC like 2% or do nothing."
    else:
        price_comparison = "Tomorrow's price is predicted to remain the same."

    # Pass data to the template
    return render_template('index.html', 
                           current_price=current_price, 
                           tomorrow_price=tomorrow_price, 
                           five_day_prices_with_index=five_day_prices_with_index,
                           price_comparison=price_comparison,
                           recommendation=recommendation)

@app.route('/get_predictions')
def get_predictions():
    # ... (your existing code to fetch data, predict, etc.)

    return {
        'current_price': current_price,
        'tomorrow_price': tomorrow_price
    } 



import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


#1. General setup
##  1.1: Download stock data
'''
    Required input:
     - ticker code
     - start date
     - end date
    Response: historical price data of the stock
'''
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    hist.reset_index(inplace=True)
    return hist

## 1.2: Feature engineering
'''
    Required input:
     - historical price data of the stock price
    Response:
     - The data set after feature engineering
'''
def feature_engineering(data,feature_list):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.dayofweek  # Add weekday as a feature
    data['Month'] = data['Date'].dt.month

    #a) Feature - MA5 
    if 'MA_5' in feature_list:
        data['MA_5'] = data['Close'].rolling(window=5).mean()
    #b) Feature - MA10
    if 'MA_10' in feature_list:
        data['MA_10'] = data['Close'].rolling(window=10).mean()

    #c) Feature - Momentum indicators （the price changed over past 5 days)
    if 'Momentum_5' in feature_list:
        data['Momentum_5'] = data['Close'] - data['Close'].shift(5)

    #d) Feature - Daily price volatility %(High-Low difference rate)
    if 'Volatility' in feature_list:
        data['Volatility'] = (data['High'] - data['Low']) / data['Low']

    #e) Feature - Daily trading volume
    if 'Total_Trade_Amount' in feature_list:
        data['Total_Trade_Amount'] = data['Volume'] * data['Close']

    #f) Feature - Relative Strength Index (RSI)
    if 'RSI' in feature_list:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

    #g) Feature - Add a placeholder for prediction correction
    if 'Prediction_Correction' in feature_list:
        data['Prediction_Correction'] = 0.0

    # Drop NaN values caused by rolling calculations
    data = data.dropna()
    return data

#2. Prediction
def prediction(method,ticker, start_date, end_date,feature_list):
    future_date = ''
    predicted_price = 0
    rmse = 0
    current_price = 0
    if method == 'xgboost':
        future_date,predicted_price,rmse,current_price = xgboost_prediction(ticker, start_date, end_date,feature_list)
    return future_date,predicted_price,rmse,current_price
##2.1 xgboost lib
'''
    Mechanism
'''
def xgboost_prediction(ticker, start_date, end_date,feature_list):
    #2.1.1) get stock price data for prediction
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    current_price = stock_data['Close'].iloc[-1]
    
    #2.1.2) process data, setup the features
    # Features:             X
    # Depandent variable:   Y
    processed_data = feature_engineering(stock_data,feature_list)
    feature_list.insert(0,'Day')
    feature_list.insert(1,'Month')
    X = processed_data[feature_list]
    y = processed_data['Close']
    
    #2.1.3) split data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42)
    
    #2.1.4) fit model
    model.fit(X_train, y_train)

    #2.1.5) Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")

    #2.1.6) Predict future value of next day
    future_date = processed_data['Date'].iloc[-1] + timedelta(days=1)
    future_date = future_date.strftime('%Y-%m-%d')
    future_features = []
    ## current latest values
    date = processed_data['Date'].iloc[-1] + timedelta(days=1)
    day = date.dayofweek
    month = date.month
    feature_value = []
    feature_value.append(day)
    feature_value.append(month)
    
    
    if 'MA_5' in feature_list:
        print('has MA_5')
        ma_5 = processed_data['Close'].iloc[-5:].mean()
        feature_value.append(ma_5)
    if 'MA_10' in feature_list:
        ma_10 = processed_data['Close'].iloc[-10:].mean()
        feature_value.append(ma_10)
    if 'Momentum_5' in feature_list:
        momentum = processed_data['Close'].iloc[-1] - processed_data['Close'].iloc[-5]
        feature_value.append(momentum)
    if 'Volatility' in feature_list:
        volatility = (processed_data['High'].iloc[-1] - processed_data['Low'].iloc[-1]) / processed_data['Low'].iloc[-1]
        feature_value.append(volatility)
    if 'Total_Trade_Amount' in feature_list:
        total_trade_amount = processed_data['Volume'].iloc[-1] * processed_data['Close'].iloc[-1]
        feature_value.append(total_trade_amount)
    if 'RSI' in feature_list:
        rsi = processed_data['RSI'].iloc[-1]
        feature_value.append(rsi)
        
    future_features.append(feature_value)
    print(feature_list)
    print(future_features)
    future_features_df = pd.DataFrame(future_features, columns=feature_list)
    predicted_price = model.predict(future_features_df.iloc[-1:].values)[0]  

    return future_date,predicted_price,rmse,current_price

# Fund Flow Function
def get_fund_flow(etfs, start_date, end_date):
    data = pd.DataFrame({})
    for description, ticker in etfs.items():
        result = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data[description] = result['Close'] * result['Volume']
    data = data.dropna()
    data.index = data.index.strftime('%Y-%m-%d')
    return data



#test
ticker = 'TSLA'
start_time = '2024-08-01'
end_time = datetime.now().strftime("%Y-%m-%d")
feature_list = [
    'MA_5', 
    'MA_10', 
    'Momentum_5', 
    'Volatility', 
    'Total_Trade_Amount',
    'RSI'
]

future_date,predicted_price,rmse,current_price = xgboost_prediction(ticker,start_time,end_time,feature_list)
print(future_date)
print(predicted_price)
print(current_price)
        


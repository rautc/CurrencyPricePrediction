import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.graph_objs as go
import numpy as np
import xgboost as xgb
import ta

# Streamlit app
st.title('Currency Price Prediction')

# User inputs
ticker = st.text_input('Enter stock ticker (e.g., AAPL, GOOG, USDINR=X):', 'USDINR=X')
predict_duration = st.number_input('Enter number of days for prediction:', min_value=1, value=10)
model_choice = st.selectbox('Choose the ML model:', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'SVM', 'XGBoost'])

# Constants for date range
start_date = datetime(2010, 1, 1)
end_date = datetime.now().date()

# Fetch historical data from Yahoo Finance
if st.button('Predict'):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    if data.empty:
        st.write('No data found for the given ticker symbol.')
    else:
        # Preprocess data
        data['Date'] = data.index
        data = data[['Date', 'Close', 'Volume']].dropna()
        data.reset_index(drop=True, inplace=True)

        # Feature Engineering: Adding technical indicators
        data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd(data['Close'])
        data['BB_High'] = ta.volatility.bollinger_hband(data['Close'])
        data['BB_Low'] = ta.volatility.bollinger_lband(data['Close'])
        data.dropna(inplace=True)

        # Create lagged features
        data['Close_Lag1'] = data['Close'].shift(1)
        data.dropna(inplace=True)

        # Features and labels
        feature_columns = ['Close_Lag1', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low']
        X = data[feature_columns].values
        y = data['Close'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the chosen model
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Ridge Regression':
            model = Ridge(alpha=1.0)
        elif model_choice == 'Lasso Regression':
            model = Lasso(alpha=0.1)
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == 'SVM':
            model = SVR()
        elif model_choice == 'XGBoost':
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Calculate accuracy metrics on the test set
        y_test_pred = model.predict(X_test)

        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        st.write(f"Model Performance on Test Data ({model_choice}):")
        st.write(f"Mean Absolute Error (MAE): {mae_test:.6f}")
        st.write(f"Mean Squared Error (MSE): {mse_test:.6f}")
        st.write(f"R-squared (R²) Score: {r2_test*100:.6f}%")

        # Predict the next n days' closing prices
        last_price = data['Close'].iloc[-1]
        last_sma_10 = data['SMA_10'].iloc[-1]
        last_ema_10 = data['EMA_10'].iloc[-1]
        last_rsi = data['RSI'].iloc[-1]
        last_macd = data['MACD'].iloc[-1]
        last_bb_high = data['BB_High'].iloc[-1]
        last_bb_low = data['BB_Low'].iloc[-1]

        predicted_dates = []
        predicted_prices = []

        for i in range(predict_duration):
            predicted_price = model.predict([[last_price, last_sma_10, last_ema_10, last_rsi, last_macd, last_bb_high, last_bb_low]])[0]
            next_date = data['Date'].iloc[-1] + timedelta(days=i+1)
            
            predicted_dates.append(next_date)
            predicted_prices.append(predicted_price)
            
            # Update features for the next iteration
            last_price = predicted_price
            last_sma_10 = (last_sma_10 * 9 + last_price) / 10  # Simplified SMA calculation
            last_ema_10 = (last_price - last_ema_10) * (2 / (10 + 1)) + last_ema_10  # EMA calculation
            last_rsi = ta.momentum.rsi(pd.Series(np.append(data['Close'].values, predicted_prices)), window=14).iloc[-1]
            last_macd = ta.trend.macd(pd.Series(np.append(data['Close'].values, predicted_prices))).iloc[-1]
            last_bb_high = ta.volatility.bollinger_hband(pd.Series(np.append(data['Close'].values, predicted_prices))).iloc[-1]
            last_bb_low = ta.volatility.bollinger_lband(pd.Series(np.append(data['Close'].values, predicted_prices))).iloc[-1]

        # Create a dataframe for the predicted prices
        predicted_df = pd.DataFrame({'Date': predicted_dates, 'Predicted_Price': predicted_prices})

        # Plot the actual prices and the predicted prices using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines+markers', name='Actual Price'))
        fig.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted_Price'], mode='lines+markers', name='Predicted Price', line=dict(dash='dash', color='red')))
        
        fig.update_layout(
            title=f'{ticker} Price Prediction for Next {predict_duration} Days',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            hovermode='x unified',
            template='plotly_white'
        )

        # Update x-axis to show only dates
        fig.update_xaxes(
            showline=True,
            showgrid=False,
            showticklabels=True,
            ticks='',
            title_text='',  # Remove title text
            tickformat='%Y-%m-%d',
            tickangle=45
        )
        
        # Remove y-axis title and grid lines for a cleaner look
        fig.update_yaxes(
            title_text='',
            showgrid=False
        )

        st.plotly_chart(fig)

        # Display the predicted prices for the next n days
        for date, price in zip(predicted_df['Date'], predicted_df['Predicted_Price']):
            st.write(f"Predicted price for {date}: {price:.2f}")

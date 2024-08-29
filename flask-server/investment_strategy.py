import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import joblib

# List of Indian stock tickers, mutual funds, gold ETFs, and bonds with diverse sectors and risk profiles
investment_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
    "HINDUNILVR.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "M&M.NS", "SUNPHARMA.NS", "HCLTECH.NS", "ONGC.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIGREEN.NS", "DMART.NS",
    # Mutual Funds
    "0P0000YENW.BO", "0P0000ZG0G.BO", "0P0000YIV3.BO",
    # Gold
    "GLD",
    # Bonds
    "TLT"
]

model_dir = "saved_models"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Fetch stock data
def fetch_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")  # Last 5 years of data
        if not hist.empty:
            model_path = os.path.join(model_dir, f"{ticker}_lstm.pkl")
            scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
            else:
                data = hist[['Close']].values
                scaler = MinMaxScaler(feature_range=(0,1))
                data_scaled = scaler.fit_transform(data)
                X_train = []
                y_train = []
                for i in range(60, len(data_scaled)):
                    X_train.append(data_scaled[i-60:i, 0])
                    y_train.append(data_scaled[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(LSTM(units=50))
                model.add(Dense(1))

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=1, batch_size=32)

                # Save the model and scaler
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)

            # Predict future prices
            data = scaler.transform(hist[['Close']])
            X_test = []
            for i in range(60, len(data)):
                X_test.append(data[i-60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_stock_price = model.predict(X_test)
            predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
            hist['Predicted'] = np.nan
            hist.iloc[60:, hist.columns.get_loc('Predicted')] = predicted_stock_price.flatten()   

            # Calculate annual return and volatility based on predicted prices
            annual_return = hist['Predicted'].pct_change().mean() * 252 * 100
            volatility = hist['Predicted'].pct_change().std() * np.sqrt(252) * 100
            beta = stock.info.get('beta', 1)
            sharpe_ratio = annual_return / volatility
            print(volatility,ticker)
            risk_profile = 'Low' if volatility < 7 else 'Medium' if volatility < 10 else 'High'

            stock_data.append({
                'Stock Name': ticker,
                'Annual Return (%)': annual_return,
                'Volatility (%)': volatility,
                'Beta': beta,
                'Sharpe Ratio': sharpe_ratio,
                'Risk Profile': risk_profile
            })
    return pd.DataFrame(stock_data)

# # User inputs
# retirement_age = int(input("Enter your retirement age: "))
# current_age = int(input("Enter your current age: "))
# desired_retirement_fund = float(input("Enter your desired retirement fund (in INR): "))
# monthly_investment = float(input("Enter how much you can invest monthly (in INR): "))

# # User chooses risk category
# print("Choose your investment category:")
# print("1. High Risk-High Return")
# print("2. Medium Risk-Medium Return")
# print("3. Low Risk-Low Return")
# risk_category = int(input("Enter the number corresponding to your choice: "))

# Map user input to risk category
# if risk_category == 1:
#     risk_category_str = "High Risk-High Return"
# elif risk_category == 2:
#     risk_category_str = "Medium Risk-Medium Return"
# elif risk_category == 3:
#     risk_category_str = "Low Risk-Low Return"
# else:
#     print("Invalid choice.")
#     exit()

# Calculate the time period
# years_to_invest = retirement_age - current_age

# Function to determine investment allocation
def suggest_investment(df, years, target_fund, monthly_investment, risk_category):
    df = df.copy()

    if risk_category == "high":
        df = df[(df['Annual Return (%)'] > 20) & (df['Volatility (%)'] > 10)]
    elif risk_category == "medium":
        df = df[(df['Annual Return (%)'] > 10) & (df['Annual Return (%)'] <= 20) & (df['Volatility (%)'] <= 10) & (df['Volatility (%)'] >= 7)]
    elif risk_category == "low":
        df = df[(df['Annual Return (%)'] <= 10) & (df['Volatility (%)'] < 7)]
    else:
        return "Invalid risk category selected."

    if df.empty:
        return "No stocks available in the selected risk category."

    df['Priority Score'] = df['Sharpe Ratio'] / df['Volatility (%)']
    df = df.sort_values(by='Priority Score', ascending=False)

    n = years * 12
    P = monthly_investment
    target = target_fund
    required_annual_return = ((target / (P * ((1 + 0.01)*n - 1) / 0.01)) * (1/n)) - 1
    required_annual_return = required_annual_return * 12

    allocation = []
    for index, row in df.iterrows():
        if row['Annual Return (%)'] >= required_annual_return:
            allocation.append((row['Stock Name'], row['Annual Return (%)'], row['Risk Profile']))
        if len(allocation) >= 10:
            break

    if not allocation:
        best_stock = df.iloc[0]
        best_annual_return = best_stock['Annual Return (%)']
        future_value = P * ((((1 + best_annual_return / 100 / 12)**n) - 1) / (best_annual_return / 100 / 12))
        return f"It's not possible to achieve your retirement goal. The maximum possible future value with the best available stock ({best_stock['Stock Name']}) is {future_value:.2f} INR."

    future_values = []
    total_future_value = 0
    total_invested = P * n
    for stock, return_rate, risk in allocation:
        future_value = P * ((((1 + return_rate / 100 / 12)**n) - 1) / (return_rate / 100 / 12))
        future_values.append((stock, return_rate, risk, future_value))
    
    total_priority_score = sum(df.loc[df['Stock Name'].isin([stock for stock, _, _ in allocation]), 'Priority Score'])
    investment_percentages = [(stock, (df.loc[df['Stock Name'] == stock, 'Priority Score'].values[0] / total_priority_score) * 100) for stock, _, _ in allocation]
    for stock, percentage in investment_percentages:
        for future_stock, return_rate, risk, future_value in future_values:
            if stock == future_stock:
                total_future_value += future_value * (percentage / 100)
                break
    # Normalize the investment percentages to sum to 100%
    
    total_percentage = sum(percentage for _, percentage in investment_percentages)
    investment_percentages = [(stock, percentage / total_percentage * 100) for stock, percentage in investment_percentages]
    
    total_return_percentage = ((total_future_value - total_invested) / total_invested) * 100
    average_annual_return_percentage = (total_return_percentage / years)

    # Monte Carlo Simulation to estimate risk
    simulations = 100000
    final_values = []
    for _ in range(simulations):
        simulated_value = 0
        for stock, return_rate, _, _ in future_values:
            
            volatility = df.loc[df['Stock Name'] == stock, 'Volatility (%)'].values[0] / 100
            simulated_annual_returns = np.random.normal(return_rate / 100, volatility, years)
            simulated_monthly_returns = (1 + simulated_annual_returns) ** (1/12) - 1
            # Filter out invalid returns
            simulated_monthly_returns = simulated_monthly_returns[simulated_monthly_returns > -1]
            #((((1 + return_rate / 100 / 12)**n) - 1) / (return_rate / 100 / 12))
            future_value = P * ((((1 + simulated_annual_returns / 100 / 12)**n) - 1) / (simulated_annual_returns / 100 / 12))
            simulated_value += future_value
        final_values.append(simulated_value)

    final_values = np.array(final_values)
    probability_of_not_achieving_goal = np.mean(final_values < target_fund) * 100

    return {
        'Total Required Annual Return (%)': required_annual_return * 100,
        'Total Future Value': total_future_value,
        'Total Invested': total_invested,
        'Total Return Percentage': total_return_percentage,
        'Average Annual Return Percentage': average_annual_return_percentage,
        'Investment Allocation': [(stock, return_rate, risk, future_value) for stock, return_rate, risk, future_value in future_values],
        'Investment Percentages': investment_percentages,
        'Probability of Not Achieving Goal (%)': probability_of_not_achieving_goal
    }

# Fetch the stock data
# df_stocks = fetch_stock_data(investment_tickers)

# # Calculate the investment strategy
# investment_strategy = suggest_investment(df_stocks, years_to_invest, desired_retirement_fund, monthly_investment, risk_category_str)

# # Display the result
# if isinstance(investment_strategy, str):
#     print(investment_strategy)
# else:
#     print(f"\nTo achieve your retirement goal, you need an annual return of approximately {investment_strategy['Total Required Annual Return (%)']:.2f}%")
#     print("Suggested investment allocation:")
#     for stock, annual_return, risk_profile, future_value in investment_strategy['Investment Allocation']:
#         print(f"Stock: {stock}, Annual Return: {annual_return:.2f}%, Risk Profile: {risk_profile}, Future Value: {future_value:.2f} INR")

#     print("\nInvestment Percentages:")
#     for stock, percentage in investment_strategy['Investment Percentages']:
#         print(f"Stock: {stock}, Percentage of Monthly Investment: {percentage:.2f}%")

#     print(f"\nCongratulations! You can achieve your retirement goal. The combined future value of your investment will be approximately {investment_strategy['Total Future Value']:.2f} INR.")
    
#     print(f"\nTotal Invested: {investment_strategy['Total Invested']:.2f} INR")
#     print(f"Total Return Percentage: {investment_strategy['Total Return Percentage']:.2f}%")
#     print(f"Average Annual Return Percentage: {investment_strategy['Average Annual Return Percentage']:.2f}%")
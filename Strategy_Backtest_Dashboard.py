import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px


from scipy.stats import norm
from datetime import date

# Title
st.title("Stock Analysis & Backtesting App")

# Timeframe Selector
interval_map = {
    "1 Minute": "1m",
    "2 Minutes": "2m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "4 Hour": "4h",
    "1 Hour": "1h",
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo"
}

# Sidebar inputs
tickers_input = st.sidebar.text_input("Enter Stock Ticker Symbols (comma-separated)", "BTC-USD,EURUSD=X,GC=F")
end_date = st.sidebar.date_input("End Date", value=date.today())
start_date = st.sidebar.date_input("Start Date", value=end_date - pd.Timedelta(days=7))
selected_tf = st.sidebar.selectbox("Select Timeframe", list(interval_map.keys()))
num_simulations = st.sidebar.slider("Monte Carlo Simulations", 10, 100, 10)
time_horizon = st.sidebar.slider("Time Horizon (Days)", 30, 365, 30)
interval = interval_map[selected_tf]

# Download data
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# Create a tab for each ticker
tabs = st.tabs(tickers)

for i, ticker in enumerate(tickers):
    with tabs[i]:
        st.header(f"📊 Results for {ticker}")

        # --- Load Data ---
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            st.error(f"⚠️ No data found for {ticker}. Check the ticker or date range.")
            continue

        data['Return'] = data['Close'].pct_change()
        data['Prev_Return'] = data['Return'].shift(1)
        ret = data['Return'].dropna()
        prev_ret = data['Prev_Return'].dropna()
        data.dropna(inplace=True)

        # Use 'Close' instead of 'Adj Close'
        data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)
        
        # --- Train Model ---
        from sklearn.linear_model import LinearRegression
        X = data[['Prev_Return']].values
        y = data['Return'].values
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        data['Predicted_Return'] = model.predict(X)
        pred = data['Predicted_Return'].dropna()

        # --- Adaptive Logic Based on Slope ---
        if slope > 0:
            logic_type = "📊 Momentum Strategy (Buy after up day)"
            data['Signal'] = np.where(data['Prev_Return'] > 0, 1, -1)
        else:
            logic_type = "🔄 Mean-Reversion Strategy (Buy after down day)"
            data['Signal'] = np.where(data['Prev_Return'] < 0, 1, -1)

        # --- Strategy Return ---
        data['Strategy_Return'] = data['Signal'] * data['Return']

        # --- Output Info ---
        st.subheader("📈 Model Summary")
        st.write(f"**Slope**: {slope:.5f}  \n**Intercept**: {intercept:.5f}")
        st.markdown(f"**Applied Strategy Logic:** {logic_type}")

        # --- Plot 1: Interactive Scatter ---
        st.subheader("📉 Scatter Plot: Yesterday vs Today Return (Interactive)")
        scatter_fig = px.scatter(
            data,
            x=data['Prev_Return'],
            y=data['Return'],
            opacity=0.5,
            labels={"Prev_Return": "Yesterday's Return", "Return": "Today's Return"},
            title="Yesterday vs Today Return"
        )
        # Add regression line
        # Align x and y for regression line to have the same length/index
        regression_x = data['Prev_Return']
        regression_y = data['Predicted_Return']
        scatter_fig.add_trace(
            go.Scatter(
                x=regression_x,
                y=regression_y,
                mode='lines',
                name='Regression Line',
                line=dict(color='red')
            )
        )
        scatter_fig.update_layout(xaxis_title="Yesterday's Return", yaxis_title="Today's Return")
        st.plotly_chart(scatter_fig, use_container_width=True)

        # --- Plot 2: Cumulative Return Comparison (Interactive) ---
        st.subheader("📊 Strategy vs Market Cumulative Return (Interactive)")
        cum_returns = data[['Return', 'Strategy_Return']].cumsum()
        cum_returns.columns = ['Market Cumulative Return', 'Strategy Cumulative Return']
        fig2 = px.line(
            cum_returns,
            labels={"value": "Cumulative Return", "index": "Date", "variable": "Type"},
            title=f"{ticker} - Strategy vs. Buy-and-Hold"
        )
        fig2.update_layout(yaxis_title="Cumulative Return", xaxis_title="Date", legend_title="Return Type")
        st.plotly_chart(fig2, use_container_width=True)

        # Monte Carlo Simulation (Interactive)
        st.subheader("Monte Carlo Simulation (Interactive)")

        last_price = data['Close'].iloc[-1]
        mean_return = data['Log Return'].mean()
        std_return = data['Log Return'].std()

        simulations = np.zeros((time_horizon, num_simulations))

        for j in range(num_simulations):
            prices = np.zeros(time_horizon)
            prices[0] = last_price
            for t in range(1, time_horizon):
                shock = np.random.normal(loc=mean_return, scale=std_return)
                prices[t] = prices[t - 1] * np.exp(shock)
            simulations[:, j] = prices

        # Create interactive plot with Plotly
        mc_fig = go.Figure()
        for j in range(num_simulations):
            mc_fig.add_trace(
            go.Scatter(
                x=np.arange(time_horizon),
                y=simulations[:, j],
                mode='lines',
                line=dict(width=1),
                opacity=0.5,
                showlegend=False
            )
            )
        mc_fig.update_layout(
            title=f"{ticker} Monte Carlo Simulation",
            xaxis_title="Day",
            yaxis_title="Price",
            template="plotly_white"
        )
        st.plotly_chart(mc_fig, use_container_width=True)

        # Black-Scholes Option Pricing
        st.subheader("Black-Scholes Option Pricing")

        # Inputs
        S = last_price
        K = st.number_input(f"Strike Price for {ticker}", value=float(round(S, 2)))
        T = st.number_input(f"Time to Maturity (Years) for {ticker}", value=0.5)
        r = st.number_input(f"Risk-Free Rate (Annual %) for {ticker}", value=5.0) / 100
        sigma = std_return * np.sqrt(252)  # annualized volatility

        # Black-Scholes formulas
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call

        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put

        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)

        st.write(f"**Call Option Price:** ${float(call_price):.2f}")
        st.write(f"**Put Option Price:** ${float(put_price):.2f}")

   
    

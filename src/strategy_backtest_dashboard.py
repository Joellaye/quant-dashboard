def run_app():
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    import plotly.io as pio
    
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
    
    for ticker in tickers:
        st.header(f"📊 Results for {ticker}")
    
        # --- Load Data ---
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        show = data.tail(25)
        
        if data.empty:
            st.error(f"⚠️ No data found for {ticker}. Check the ticker or date range.")
            continue
    
        data['Return'] = data['Close'].pct_change()
        data['Prev_Return'] = data['Return'].shift(1)
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
    
        # --- Plot 1: Scatter ---
        st.subheader("📉 Scatter Plot: Yesterday vs Today Return")
        fig1, ax1 = plt.subplots()
        ax1.scatter(data['Prev_Return'], data['Return'], alpha=0.5, label="Actual")
        ax1.plot(data['Prev_Return'], data['Predicted_Return'], color='red', label="Regression Line")
        ax1.set_xlabel("Yesterday's Return")
        ax1.set_ylabel("Today's Return")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)
    
        # --- Plot 2: Cumulative Return Comparison ---
        st.subheader("📊 Strategy vs Market Cumulative Return")
        fig2, ax2 = plt.subplots()
        data[['Return', 'Strategy_Return']].cumsum().plot(ax=ax2)
        ax2.set_ylabel("Cumulative Return")
        ax2.set_title(f"{ticker} - Strategy vs. Buy-and-Hold")
        ax2.grid(True)
        st.pyplot(fig2)


        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation")
    
        last_price = data['Close'].iloc[-1]
        mean_return = data['Log Return'].mean()
        std_return = data['Log Return'].std()
    
        simulations = np.zeros((time_horizon, num_simulations))
    
        for i in range(num_simulations):
            prices = np.zeros(time_horizon)
            prices[0] = last_price
            for t in range(1, time_horizon):
                shock = np.random.normal(loc=mean_return, scale=std_return)
                prices[t] = prices[t - 1] * np.exp(shock)
            simulations[:, i] = prices
    
        fig, ax = plt.subplots()
        ax.plot(simulations)
        ax.set_title(f"{ticker} Monte Carlo Simulation")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        st.pyplot(fig)
    
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
    
       
        

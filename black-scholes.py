import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from scipy.stats import norm

# Core class for Black-Scholes Model
class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type):
        d1, d2 = self._d1_d2()
        if option_type == 'Call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'Put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def greeks(self, option_type):
        d1, d2 = self._d1_d2()
        delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) if option_type == 'Call' else (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) if option_type == 'Call' else -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

# Visualization of sensitivities and Greeks
def generate_sensitivity_plot(param_name, param_values, S, K, T, r, sigma, option_type):
    prices = []
    for val in param_values:
        kwargs = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        kwargs[param_name] = val
        model = BlackScholes(**kwargs)
        prices.append(model.price(option_type))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_values, y=prices, mode='lines', name='Option Price'))
    fig.update_layout(title=f"{option_type} Price vs {param_name.capitalize()}", xaxis_title=param_name.capitalize(), yaxis_title='Option Price')
    return fig

def generate_greeks_plot(param_name, param_values, S, K, T, r, sigma, option_type):
    fig = go.Figure()
    for greek in ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']:
        greek_values = []
        for val in param_values:
            kwargs = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
            kwargs[param_name] = val
            model = BlackScholes(**kwargs)
            greek_values.append(model.greeks(option_type)[greek])
        fig.add_trace(go.Scatter(x=param_values, y=greek_values, mode='lines', name=greek))
    fig.update_layout(title=f"Greeks vs {param_name.capitalize()} ({option_type})", xaxis_title=param_name.capitalize(), yaxis_title='Value')
    return fig

# Order Flow
@st.cache_data
def get_stock_data(ticker):
    return yf.Ticker(ticker).history(period="1y")

def show_order_flow(ticker):
    data = get_stock_data(ticker)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], yaxis='y2', name='Volume', opacity=0.4))
    fig.update_layout(title=f"Order Flow for {ticker}", yaxis=dict(title='Price'), yaxis2=dict(title='Volume', overlaying='y', side='right'), xaxis_title='Date')
    return fig

# Streamlit layout
st.set_page_config(page_title="Black-Scholes Analysis", layout="wide")
st.title("\U0001F4C8 Black-Scholes Option Analysis Tool")

col_left, col_right = st.columns(2)
with col_left:
    st.header("Define Parameters")
    option_type = st.radio("Option Type", ["Call", "Put"])
    S = st.slider("Stock Price (S)", 50.0, 200.0, 100.0)
    K = st.slider("Strike Price (K)", 50.0, 200.0, 100.0)
    T = st.slider("Time to Maturity (T in years)", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05)
    sigma = st.slider("Volatility (σ)", 0.1, 0.6, 0.2)

    model = BlackScholes(S, K, T, r, sigma)
    st.subheader("Option Prices")
    st.markdown(f"**Call Option Price:** {model.price('Call'):.2f}")
    st.markdown(f"**Put Option Price:** {model.price('Put'):.2f}")

    st.subheader("Greeks Summary")
    greeks = model.greeks(option_type)
    for greek, value in greeks.items():
        st.markdown(f"**{greek}:** {value:.4f}")

with col_right:
    st.header("Visual Analysis")
    param = st.selectbox("Select Parameter to Vary", ["S", "K", "T", "r", "sigma"])
    param_ranges = {
        "S": np.linspace(50, 150, 100),
        "K": np.linspace(50, 150, 100),
        "T": np.linspace(0.1, 2.0, 100),
        "r": np.linspace(0.0, 0.1, 100),
        "sigma": np.linspace(0.1, 0.6, 100)
    }
    st.plotly_chart(generate_sensitivity_plot(param, param_ranges[param], S, K, T, r, sigma, option_type))
    st.plotly_chart(generate_greeks_plot(param, param_ranges[param], S, K, T, r, sigma, option_type))

st.divider()
st.header("Broker Order Flow")
ticker = st.text_input("Enter stock ticker (e.g., GOOG, MSFT, SPY)", value="SPY")
st.plotly_chart(show_order_flow(ticker))

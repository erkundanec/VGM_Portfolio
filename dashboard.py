import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go


# Sector analysis plot function
def sector_analysis_plot(sectors, nifty_weights, portfolio_weights):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Positioning data
    index = np.arange(len(sectors))
    bar_width = 0.4

    # Plot Nifty and Portfolio weights
    ax.barh(index - bar_width/2, nifty_weights, bar_width, label='Nifty', color='blue', alpha=0.6)
    ax.barh(index + bar_width/2, portfolio_weights, bar_width, label='Portfolio', color='red', alpha=0.6)

    # Labels and titles
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Sector Allocation: Nifty vs Portfolio')
    ax.set_yticks(index)
    ax.set_yticklabels(sectors)

    # Adding legend
    ax.legend()

    # Save the plot to a bytes buffer and encode it as a base64 string
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image in base64
    image_base64 = base64.b64encode(image_png).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"


# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    stock_df = pd.DataFrame(stock_data)
    return stock_df


# Calculate metrics like returns and cumulative returns
def calculate_metrics(stock_df):
    returns = stock_df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    volatility = returns.std() * np.sqrt(252)
    annual_return = returns.mean() * 252
    return returns, cumulative_returns, volatility, annual_return


# Simulate a portfolio with given weights
def simulate_portfolio(stock_df, weights):
    returns = stock_df.pct_change().dropna()
    portfolio_return = np.dot(returns.mean(), weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility


# Plot stock prices
def plot_stock_prices(stock_df):
    fig = go.Figure()
    for ticker in stock_df.columns:
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df[ticker], mode='lines', name=ticker))
    fig.update_layout(title="Stock Prices", xaxis_title="Date", yaxis_title="Price")
    return fig


# Plot cumulative returns
def plot_cumulative_returns(cumulative_returns):
    fig = go.Figure()
    for ticker in cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[ticker], mode='lines', name=ticker))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
    return fig


# Plot risk and return
def plot_risk_return(annual_return, volatility):
    fig = go.Figure(data=[go.Scatter(x=volatility, y=annual_return, mode='markers', text=annual_return.index)])
    fig.update_layout(title="Risk vs Return", xaxis_title="Volatility (Risk)", yaxis_title="Annual Return")
    return fig


# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Portfolio Analysis Dashboard"),
    
    # Input fields for stock tickers and date range
    dcc.Input(id='stock-tickers', value='AAPL, MSFT, TSLA', type='text'),
    dcc.DatePickerRange(
        id='date-range',
        start_date='2020-01-01',
        end_date='2024-01-01'
    ),
    
    # Graphs to display
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='cumulative-returns-chart'),
    dcc.Graph(id='risk-return-chart'),
    
    # Sector Analysis Image
    html.Img(id='sector-analysis-chart'),

    # Display portfolio metrics
    html.Div(id='portfolio-metrics')
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('cumulative-returns-chart', 'figure'),
     Output('risk-return-chart', 'figure'),
     Output('sector-analysis-chart', 'src'),  # Update the image source
     Output('portfolio-metrics', 'children')],
    [Input('stock-tickers', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dashboard(ticker_input, start_date, end_date):
    tickers = [ticker.strip() for ticker in ticker_input.split(',')]
    
    # Fetch stock data and calculate metrics
    stock_df = fetch_stock_data(tickers, start_date, end_date)
    returns, cumulative_returns, volatility, annual_return = calculate_metrics(stock_df)
    
    # Simulate equal-weighted portfolio
    weights = np.array([1/len(tickers)] * len(tickers))
    portfolio_return, portfolio_volatility = simulate_portfolio(stock_df, weights)
    
    # Plot graphs
    price_chart = plot_stock_prices(stock_df)
    cumulative_returns_chart = plot_cumulative_returns(cumulative_returns)
    risk_return_chart = plot_risk_return(annual_return, volatility)
    
    # Sector analysis data (replace with actual sector data for your portfolio)
    sectors = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 
               'Financials', 'Health Care', 'Industrials', 'Utilities', 'Information Technology', 'Materials']
    nifty_weights = [2.59, 6.74, 10.04, 14.24, 28.19, 3.6, 3.0, 1.83, 20.82, 8.97]  # Nifty sector weights
    portfolio_weights = [2.5, 5.5, 9.0, 12.5, 35, 19.5, 12.3, 0, 3, 0]  # Portfolio sector weights

    # Generate the sector analysis plot
    sector_image = sector_analysis_plot(sectors, nifty_weights, portfolio_weights)
    
    # Portfolio metrics summary
    portfolio_metrics = f"Portfolio Annual Return: {portfolio_return:.2%}, Portfolio Volatility: {portfolio_volatility:.2%}"
    
    return price_chart, cumulative_returns_chart, risk_return_chart, sector_image, portfolio_metrics

if __name__ == '__main__':
    app.run_server(debug=True)

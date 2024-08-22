import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
import dash_bootstrap_components as dbc


# Black-Scholes Model Functions
def blackScholes(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def optionDelta(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if type == "c":
        delta = norm.cdf(d1)
    elif type == "p":
        delta = -norm.cdf(-d1)
    return delta

def optionGamma(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def optionTheta(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta / 365

def optionVega(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
    return vega

def optionRho(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        rho = 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        rho = 0.01 * -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return rho
# Binomial Options Pricing Model
def binomial_option_pricing(S, K, r, T, sigma, N, type="c"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    ST = np.zeros(N + 1)
    for i in range(N + 1):
        ST[i] = S * (u ** (N - i)) * (d ** i)
    
    option_values = np.zeros(N + 1)
    if type == "c":
        option_values = np.maximum(0, ST - K)
    elif type == "p":
        option_values = np.maximum(0, K - ST)
    
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = np.exp(-r * dt) * (q * option_values[i] + (1 - q) * option_values[i + 1])
    
    return option_values[0]

# Monte Carlo Simulation
def monte_carlo_option_pricing(S, K, r, T, sigma, M, N, type="c"):
    dt = T / N
    payoff = np.zeros(M)

    for i in range(M):
        ST = S
        for j in range(N):
            ST *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())

        if type == "c":
            payoff[i] = max(0, ST - K)
        elif type == "p":
            payoff[i] = max(0, K - ST)

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Initialize the Dash app with Bootstrap for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Option Pricing Models", className='text-center my-4'),
    
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Label('Underlying Asset Price (₹)'), width=12),
                dbc.Col(dcc.Input(id='S-input', value=25000, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Strike Price (₹)'), width=12),
                dbc.Col(dcc.Input(id='K-input', value=25500, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Risk-Free Rate (%)'), width=12),
                dbc.Col(dcc.Input(id='r-input', value=6.5, type='number', step=0.01, className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Volatility (%)'), width=12),
                dbc.Col(dcc.Input(id='sigma-input', value=25, type='number', step=0.01, className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Time to Expiry (days)'), width=12),
                dbc.Col(dcc.Input(id='T-input', value=7, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Number of Steps (for Binomial & Monte Carlo)'), width=12),
                dbc.Col(dcc.Input(id='N-input', value=10, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Number of Simulations (for Monte Carlo)'), width=12),
                dbc.Col(dcc.Input(id='M-input', value=100, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Option Type'), width=12),
                dbc.Col(dcc.Dropdown(
                    id='option-type',
                    options=[
                        {'label': 'Call', 'value': 'c'},
                        {'label': 'Put', 'value': 'p'}
                    ],
                    value='c',
                    className='form-control'
                ), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Pricing Model'), width=12),
                dbc.Col(dcc.Dropdown(
                    id='model-type',
                    options=[
                        {'label': 'Black-Scholes', 'value': 'bs'},
                        {'label': 'Binomial', 'value': 'binomial'},
                        {'label': 'Monte Carlo', 'value': 'monte_carlo'}
                    ],
                    value='bs',
                    className='form-control'
                ), width=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sensitivity-graph')),
            ])

        ], width=3),
        
        dbc.Col([
            html.Div(id='option-metrics', className='mb-4'),
            dcc.Graph(id='price-graph'),
            dcc.Graph(id='delta-graph'),
            dcc.Graph(id='gamma-graph'),
            dcc.Graph(id='theta-graph'),
            dcc.Graph(id='vega-graph'),
            dcc.Graph(id='rho-graph'),
        ], width=9),
    ]),
], fluid=True)

@app.callback(
    [Output('option-metrics', 'children'),
     Output('price-graph', 'figure'),
     Output('delta-graph', 'figure'),
     Output('gamma-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('vega-graph', 'figure'),
     Output('rho-graph', 'figure'),
     Output('sensitivity-graph', 'figure')],
    [Input('S-input', 'value'),
     Input('K-input', 'value'),
     Input('r-input', 'value'),
     Input('sigma-input', 'value'),
     Input('T-input', 'value'),
     Input('N-input', 'value'),
     Input('M-input', 'value'),
     Input('option-type', 'value'),
     Input('model-type', 'value')]
)
def update_graphs(S, K, r, sigma, T, N, M, type_input, model_type):
    S = float(S)
    K = float(K)
    r = float(r) / 100
    sigma = float(sigma) / 100
    T = float(T) / 365
    
    if model_type == 'bs':
        price = blackScholes(S, K, r, T, sigma, type=type_input)
        delta = optionDelta(S, K, r, T, sigma, type=type_input)
        gamma = optionGamma(S, K, r, T, sigma)
        theta = optionTheta(S, K, r, T, sigma, type=type_input)
        vega = optionVega(S, K, r, T, sigma)
        rho = optionRho(S, K, r, T, sigma, type=type_input)
        
        option_metrics = [
            html.H4(f"Option Price (₹): {round(price, 2)}"),
            html.H5(f"Delta: {round(delta, 4)}"),
            html.H5(f"Gamma: {round(gamma, 4)}"),
            html.H5(f"Theta: {round(theta, 4)}"),
            html.H5(f"Vega: {round(vega, 4)}"),
            html.H5(f"Rho: {round(rho, 4)}"),
        ]
        
        x = np.linspace(S - 1000, S + 1000, 100)
        y_price = [blackScholes(xi, K, r, T, sigma, type=type_input) for xi in x]
        y_delta = [optionDelta(xi, K, r, T, sigma, type=type_input) for xi in x]
        y_gamma = [optionGamma(xi, K, r, T, sigma) for xi in x]
        y_theta = [optionTheta(xi, K, r, T, sigma, type=type_input) for xi in x]
        y_vega = [optionVega(xi, K, r, T, sigma) for xi in x]
        y_rho = [optionRho(xi, K, r, T, sigma, type=type_input) for xi in x]

        price_graph = go.Figure(data=go.Scatter(x=x, y=y_price, mode='lines', name='Option Price'))
        price_graph.update_layout(title='Option Price vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Option Price')
        
        delta_graph = go.Figure(data=go.Scatter(x=x, y=y_delta, mode='lines', name='Delta'))
        delta_graph.update_layout(title='Delta vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Delta')

        gamma_graph = go.Figure(data=go.Scatter(x=x, y=y_gamma, mode='lines', name='Gamma'))
        gamma_graph.update_layout(title='Gamma vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Gamma')

        theta_graph = go.Figure(data=go.Scatter(x=x, y=y_theta, mode='lines', name='Theta'))
        theta_graph.update_layout(title='Theta vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Theta')

        vega_graph = go.Figure(data=go.Scatter(x=x, y=y_vega, mode='lines', name='Vega'))
        vega_graph.update_layout(title='Vega vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Vega')

        rho_graph = go.Figure(data=go.Scatter(x=x, y=y_rho, mode='lines', name='Rho'))
        rho_graph.update_layout(title='Rho vs. Underlying Asset Price',
                                 xaxis_title='Underlying Asset Price',
                                 yaxis_title='Rho')

    elif model_type == 'binomial':
        price = binomial_option_pricing(S, K, r, T, sigma, N, type=type_input)
        delta = gamma = theta = vega = rho = None
        
        option_metrics = [
            html.H4(f"Option Price (₹): {round(price, 2)}")
        ]
        
        x = np.linspace(S - 1000, S + 1000, 100)
        y_price = [binomial_option_pricing(xi, K, r, T, sigma, N, type=type_input) for xi in x]
        price_graph = go.Figure(data=go.Scatter(x=x, y=y_price, mode='lines', name='Option Price'))
        price_graph.update_layout(title='Option Price vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Option Price')

        delta_graph = gamma_graph = theta_graph = vega_graph = rho_graph = go.Figure()  # Empty figures

    elif model_type == 'monte_carlo':
        price = monte_carlo_option_pricing(S, K, r, T, sigma, M, N, type=type_input)
        delta = gamma = theta = vega = rho = None
        
        option_metrics = [
            html.H4(f"Option Price (₹): {round(price, 2)}")
        ]
        
        x = np.linspace(S - 1000, S + 1000, 100)
        y_price = [monte_carlo_option_pricing(xi, K, r, T, sigma, M, N, type=type_input) for xi in x]
        price_graph = go.Figure(data=go.Scatter(x=x, y=y_price, mode='lines', name='Option Price'))
        price_graph.update_layout(title='Option Price vs. Underlying Asset Price',
                                  xaxis_title='Underlying Asset Price',
                                  yaxis_title='Option Price')

        delta_graph = gamma_graph = theta_graph = vega_graph = rho_graph = go.Figure()  # Empty figures

    # Sensitivity Graph
    x = np.linspace(0.1, 0.5, 100)  # Volatility range
    y_price = np.zeros_like(x)

    for i, s in enumerate(x):
        if model_type == 'bs':
            y_price[i] = blackScholes(S, K, r, T, s, type='c')
        elif model_type == 'binomial':
            y_price[i] = binomial_option_pricing(S, K, r, T, s, N=10, type='c')
        elif model_type == 'monte_carlo':
            y_price[i] = monte_carlo_option_pricing(S, K, r, T, s, M=100, N=10, type='c')

    sensitivity_graph = go.Figure(data=go.Scatter(x=x, y=y_price, mode='lines', name='Option Price'))
    sensitivity_graph.update_layout(title='Option Price Sensitivity to Volatility',
                                    xaxis_title='Volatility',
                                    yaxis_title='Option Price')

    return option_metrics, price_graph, delta_graph, gamma_graph, theta_graph, vega_graph, rho_graph, sensitivity_graph

if __name__ == '__main__':
    app.run_server(debug=True)
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
    dbc.Row([
        dbc.Col(
            html.Div([
                html.A(
                    [   
                        "Created by :  ",
                        html.Img(src='https://img.icons8.com/color/48/000000/linkedin.png', style={'width': '20px', 'margin-right': '10px'}),
                        "Gokul S G"
                    ], href='https://www.linkedin.com/in/sggokul/', target='_blank', className='text-decoration-none'
                )
            ], className='text-left', style={'position': 'relative', 'margin-top': '20px'}),
            width=2
        )
    ]),
    html.H1("Option Pricing Models", className='text-center my-4'),
    
    dbc.Row([
        # Left Column: Inputs
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

            # New inputs for heatmap
            dbc.Row([
                dbc.Col(dbc.Label('Min Spot Price (₹)'), width=12),
                dbc.Col(dcc.Input(id='min-spot', value=24000, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Max Spot Price (₹)'), width=12),
                dbc.Col(dcc.Input(id='max-spot', value=26000, type='number', className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Min Volatility (%)'), width=12),
                dbc.Col(dcc.Input(id='min-vol', value=20, type='number', step=0.01, className='form-control'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label('Max Volatility (%)'), width=12),
                dbc.Col(dcc.Input(id='max-vol', value=30, type='number', step=0.01, className='form-control'), width=12),
            ])
        ], md=5),
        
        # Right Column: Outputs
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Option Price", className="card-title"),
                    html.H6(id='option-price', className="card-subtitle"),
                ]),
                className="mb-4"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5("Greeks", className="card-title"),
                    html.Div(id='greeks-container')
                ]),
                className="mb-4"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sensitivity-graph')),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='heatmap-graph')),
            ])
        ], md=7),
    ])
])

# Callback to calculate option price and Greeks
@app.callback(
    [
        Output('option-price', 'children'),
        Output('greeks-container', 'children'),
        Output('sensitivity-graph', 'figure'),
        Output('heatmap-graph', 'figure'),
    ],
    [
        Input('S-input', 'value'),
        Input('K-input', 'value'),
        Input('r-input', 'value'),
        Input('sigma-input', 'value'),
        Input('T-input', 'value'),
        Input('N-input', 'value'),
        Input('M-input', 'value'),
        Input('option-type', 'value'),
        Input('model-type', 'value'),
        Input('min-spot', 'value'),
        Input('max-spot', 'value'),
        Input('min-vol', 'value'),
        Input('max-vol', 'value'),
    ]
)
def update_option_price(S, K, r, sigma, T, N, M, option_type, model_type, min_spot, max_spot, min_vol, max_vol):
    T /= 365
    r /= 100
    sigma /= 100
    option_price = None
    if model_type == 'bs':
        option_price = blackScholes(S, K, r, T, sigma, option_type)
    elif model_type == 'binomial':
        option_price = binomial_option_pricing(S, K, r, T, sigma, N, option_type)
    elif model_type == 'monte_carlo':
        option_price = monte_carlo_option_pricing(S, K, r, T, sigma, M, N, option_type)
    
    greeks_output = None
    if model_type == 'bs':
        delta = optionDelta(S, K, r, T, sigma, option_type)
        gamma = optionGamma(S, K, r, T, sigma)
        theta = optionTheta(S, K, r, T, sigma, option_type)
        vega = optionVega(S, K, r, T, sigma)
        rho = optionRho(S, K, r, T, sigma, option_type)
        greeks_output = [
            html.H6(f"Delta: {delta:.4f}", className="card-subtitle"),
            html.H6(f"Gamma: {gamma:.4f}", className="card-subtitle"),
            html.H6(f"Theta: {theta:.4f}", className="card-subtitle"),
            html.H6(f"Vega: {vega:.4f}", className="card-subtitle"),
            html.H6(f"Rho: {rho:.4f}", className="card-subtitle"),
        ]
    
    # Generate sensitivity graph
    spot_prices = np.linspace(min_spot, max_spot, 100)
    option_prices = [blackScholes(s, K, r, T, sigma, option_type) for s in spot_prices]
    sensitivity_figure = go.Figure(data=go.Scatter(x=spot_prices, y=option_prices, mode='lines', name='Option Price'))
    sensitivity_figure.update_layout(title='Option Price Sensitivity to Spot Price', xaxis_title='Spot Price', yaxis_title='Option Price')

    # Generate heatmap graph
    spot_range = np.linspace(min_spot, max_spot, 50)
    vol_range = np.linspace(min_vol/100, max_vol/100, 50)
    heatmap_data = np.array([[blackScholes(s, K, r, T, v, option_type) for s in spot_range] for v in vol_range])

    heatmap_figure = go.Figure(data=go.Heatmap(z=heatmap_data, x=spot_range, y=vol_range*100, colorscale='Viridis'))
    heatmap_figure.update_layout(title='Option Price Heatmap', xaxis_title='Spot Price', yaxis_title='Volatility (%)')

    return (f"₹{option_price:.2f}", greeks_output, sensitivity_figure, heatmap_figure)

if __name__ == '__main__':
    app.run_server(debug=True)

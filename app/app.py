import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import prediction_models as predictions
import datetime
import pandas as pd
import yfinance as yf

def get_fund_flow(etfs, start_date, end_date):
    # Create a new dataset, set the index
    data = pd.DataFrame({})
    # Get each ETF code
    for description, ticker in etfs.items():
        result = predictions.yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Get price and calculate the fund flow
        data[description] = result['Close'] * result['Volume']
    # Drop NAN
    data = data.dropna()
    data.index = data.index.strftime('%Y-%m-%d')
    return data

def fetch_sector_data(sectors, period):
    data = {}
    for sector in sectors.keys():
        etf = yf.Ticker(sector)
        hist = etf.history(period=period)  # 獲取指定時間區間的數據
        data[sector] = hist['Close']
    return pd.DataFrame(data)

# 計算 RSI 函數
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 計算動量和 RSI 強度
def calculate_indicators(data):
    indicators = {'Sector': [], 'Momentum': [], 'RSI': []}
    for sector in data.columns:
        rsi = calculate_rsi(data[sector]).iloc[-1]  # 計算 RSI 並取最後一個值
        strength = (data[sector].iloc[-1] - data[sector].iloc[0]) / data[sector].iloc[0] * 100  # 價格強度
        momentum = (data[sector].iloc[-1] - data[sector].mean()) / data[sector].mean() * 100   # 動量
        
        indicators['Sector'].append(sector)
        indicators['Momentum'].append(momentum)
        indicators['RSI'].append(rsi)
    return pd.DataFrame(indicators)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define ETF options
etfs = {
    '能源 Energy': 'XLE',
    '公用事业 Utilities': 'XLU',
    '科技 Technology': 'XLK',
    '金融 Financials': 'XLF',
    '健康 Health care': 'XLV',
    '工业 Industrials': 'XLI',
    '材料 Materials': 'XLB',
    '房地产 Real estate': 'XLRE',
    '非必需消费品 Consumer Discretionary': 'XLY',
    '必需消费品 Consumer staples': 'XLP'
}

# Define available parameters
available_parameters = [
    'MA_5',
    'MA_10',
    'Momentum_5',
    'Volatility',
    'Total_Trade_Amount',
    'RSI'
]

# Define layout
app.layout = dbc.Container([
    dbc.Row([html.H1("Dawnlight System", className="text-center my-4")]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Menu", className="mb-3"),
                dbc.Nav([
                    dbc.NavLink("Price Prediction", href="/price-prediction", id="price-prediction-link", active=True),
                    dbc.NavLink("Overall Industrial Fund Flow", href="/fund-flow", id="fund-flow-link"),
                    dbc.NavLink("Sector Rotation", href="/sector-rotation", id="sector-rotation-link")
                ], vertical=True, pills=True)
            ], className="bg-light p-3 h-100")
        ], width=2),
        dbc.Col([
            # Wrap the main content with a loading spinner
            dcc.Loading(
                id="loading-spinner",
                type="circle",  # Type of spinner (can be 'circle', 'dot', etc.)
                children=html.Div(id="main-content", className="p-4")
            )
        ], width=10)
    ])
], fluid=True)

@app.callback(
    [Output("price-prediction-link", "active"), Output("fund-flow-link", "active"),Output("sector-rotation-link", "active"), Output("main-content", "children")],
    [Input("price-prediction-link", "n_clicks"), Input("fund-flow-link", "n_clicks"),Input("sector-rotation-link", "n_clicks")]
)
def update_main_content(price_clicks, fund_clicks,sector_rotation_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == "price-prediction-link.n_clicks":
        content = html.Div([
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(
                        id="etf-dropdown",
                        options=[{"label": key, "value": value} for key, value in etfs.items()],
                        placeholder="Select an ETF",
                        className="mb-3"
                    ),
                    dbc.Input(id="stock-input", type="text", placeholder="Or enter a Stock Ticker (e.g., TSLA)", className="mb-3"),
                    dcc.Checklist(
                        id="parameter-checklist",
                        options=[{"label": param, "value": param} for param in available_parameters],
                        value=available_parameters,  # Default to all selected
                        labelStyle={"display": "block"},
                        className="mb-3"
                    ),
                    dbc.Button("Predict", id="predict-btn", color="primary", n_clicks=0)
                ])
            ], className="bg-light p-3 mb-4"),
            html.Div(id="prediction-output")
        ])
        return True, False,False, content
    elif ctx.triggered[0]["prop_id"] == "fund-flow-link.n_clicks":
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')  # 當前日期
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')  # 計算過去1個月的日期
        try:
            data = get_fund_flow(etfs, start_date, end_date)

            # Create time series plot
            line_fig = go.Figure()
            for column in data.columns:
                line_fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))

            line_fig.update_layout(
                title="ETF Fund Flow Over Time",
                xaxis_title="Date",
                yaxis_title="Fund Flow",
                template="plotly_white"
            )

            # Create bar plot for last two days
            fund_flow_last = data.iloc[-1]
            fund_flow_prev = data.iloc[-2]
            fund_flow_prev_2 = data.iloc[-3]
            fund_flow_prev_3 = data.iloc[-4]
            fund_flow_prev_4 = data.iloc[-5]

            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=fund_flow_prev_4.index,
                y=fund_flow_prev_4.values,
                name="T-4",
                marker_color='blue'
            ))

            bar_fig.add_trace(go.Bar(
                x=fund_flow_prev_3.index,
                y=fund_flow_prev_3.values,
                name="T-3",
                marker_color='pink'
            ))
            bar_fig.add_trace(go.Bar(
                x=fund_flow_prev_2.index,
                y=fund_flow_prev_2.values,
                name="T-2",
                marker_color='red'
            ))

            bar_fig.add_trace(go.Bar(
                x=fund_flow_prev.index,
                y=fund_flow_prev.values,
                name="T-1 (Previous Day)",
                marker_color='orange'
            ))
            
            bar_fig.add_trace(go.Bar(
                x=fund_flow_last.index,
                y=fund_flow_last.values,
                name="Last Day",
                marker_color='indigo'
            ))
            

            bar_fig.update_layout(
                title="Total Fund Flow by Sector (Last Two Days)",
                xaxis_title="Sector",
                yaxis_title="Fund Flow",
                template="plotly_white",
                barmode='group'
            )

            content = html.Div([
                dcc.Graph(figure=line_fig),
                dcc.Graph(figure=bar_fig)
            ])
            return False, True,False, content
        except Exception as e:
            return False, True, html.Div(f"Error: {str(e)}", style={"color": "red"})
    elif ctx.triggered[0]["prop_id"] == "sector-rotation-link.n_clicks":
        sectors = {
            'XLE': '能源',
            'XLK': '科技',
            'XLY': '非必需消費',
            'XLF': '金融',
            'XLC': '通信',
            'XLI': '工業',
            'XLU': '公用事業',
            'XLP': '必需消費',
            'XLB': '材料',
            'XLRE': '房地產'
        }

        time_period = '1mo'
        raw_data = fetch_sector_data(sectors, time_period)
        indicators = calculate_indicators(raw_data)
        fig = go.Figure()
        for i, sector in enumerate(sectors.keys()):
            fig.add_trace(go.Scatter(
                x=[indicators['RSI'][i]],  # RSI 作為 X 軸
                y=[indicators['Momentum'][i]],  # 動量作為 Y 軸
                mode='markers+text',
                name=sectors[sector],
                text=[sectors[sector]],
                textposition="top center",
                marker=dict(size=15, opacity=0.8)
            ))
        # 添加四象限背景顏色
        fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
        fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(255, 255, 0, 0.1)", layer="below", line_width=0)
        fig.add_shape(type="rect", x0=0, y0=-50, x1=50, y1=0, fillcolor="rgba(0, 255, 0, 0.1)", layer="below", line_width=0)
        fig.add_shape(type="rect", x0=50, y0=-50, x1=100, y1=0, fillcolor="rgba(0, 0, 255, 0.1)", layer="below", line_width=0)

        # 添加四個象限標註
        fig.add_annotation(x=75, y=40, text="跑贏區", showarrow=False, font=dict(size=16, color="red"))
        fig.add_annotation(x=25, y=40, text="改善區", showarrow=False, font=dict(size=16, color="orange"))
        fig.add_annotation(x=75, y=-40, text="轉弱區", showarrow=False, font=dict(size=16, color="blue"))
        fig.add_annotation(x=25, y=-40, text="跑輸區", showarrow=False, font=dict(size=16, color="green"))

        # 添加四個區域的定義文字
        fig.add_annotation(
            x=75, y=45, text="跑贏區：RSI 高且動量強，表現最佳", 
            showarrow=False, font=dict(size=12, color="red")
        )
        fig.add_annotation(
            x=25, y=45, text="改善區：RSI 低但動量強，表現改善", 
            showarrow=False, font=dict(size=12, color="orange")
        )
        fig.add_annotation(
            x=75, y=-45, text="轉弱區：RSI 高但動量弱，表現轉弱", 
            showarrow=False, font=dict(size=12, color="blue")
        )
        fig.add_annotation(
            x=25, y=-45, text="跑輸區：RSI 低且動量弱，表現最差", 
            showarrow=False, font=dict(size=12, color="green")
        )

        # 設置圖表標題和軸範圍
        fig.update_layout(
            title="美股板塊強弱輪動圖 (RSI vs 動量強度) - 過去1個月",
            xaxis_title="RSI (0-100)",
            yaxis_title="動量強度 (%)",
            xaxis=dict(range=[0, 100], zeroline=True),  # RSI 範圍 0-100
            yaxis=dict(range=[-50, 50], zeroline=True),  # 動量範圍
            template="plotly_dark"
        )
        content = html.Div([
                dcc.Graph(figure=fig)
        ])
        return False, False,True, content


@app.callback(
    [Output("prediction-output", "children"), Output("stock-input", "value")],
    [Input("predict-btn", "n_clicks")],
    [State("stock-input", "value"), State("etf-dropdown", "value"), State("parameter-checklist", "value")]
)
def update_prediction(n_clicks, stock_input, etf_value, selected_parameters):
    if n_clicks > 0 and (stock_input or etf_value):
        ticker = stock_input.strip().upper() if stock_input else etf_value
        try:
            # Fetch historical data and predictions
            historical_data = predictions.fetch_stock_data(ticker, '2024-08-01', datetime.datetime.now().strftime("%Y-%m-%d"))
            future_date, future_predictions, rmse, current_price = predictions.prediction(
                'xgboost',
                ticker,
                '2024-08-01',
                datetime.datetime.now().strftime("%Y-%m-%d"),
                selected_parameters
            )

            # Prepare data for the plot
            historical_dates = historical_data['Date']
            historical_prices = historical_data['Close']

            fig = go.Figure()

            # Add historical prices line
            fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices', line=dict(color='blue')))

            # Add line connecting last historical price to predicted price
            fig.add_trace(go.Scatter(
                x=[historical_dates.iloc[-1], future_date],
                y=[historical_prices.iloc[-1], future_predictions],
                mode='lines',
                name='Prediction Transition',
                line=dict(color='green', dash='dash')
            ))

            # Add predicted price point
            fig.add_trace(go.Scatter(x=[future_date], y=[future_predictions], mode='markers', name='Predicted Price', marker=dict(color='orange', size=10)))

            fig.update_layout(
                title=f"{ticker.upper()} Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )

            evaluation_text = f"RMSE: {rmse:.2f}\nCurrent price: {current_price:.2f}\nNext trading date price: {future_predictions:.2f} ({(future_predictions-current_price)/current_price*100:.2f}%)"
            return html.Div([html.Pre(evaluation_text),dcc.Graph(figure=fig)]), stock_input  # Keep the input value
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={"color": "red"}), stock_input
    return "", stock_input

if __name__ == "__main__":
    app.run_server(debug=True)

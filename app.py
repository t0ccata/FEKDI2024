from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Load the model once when the app starts
model = joblib.load('weekly_prophet_model.pkl')

def create_future_dataframe(start_date, periods):
    return pd.date_range(start=start_date, periods=periods, freq='W-SUN').to_frame(index=False, name='ds')

def make_forecast(model, future):
    return model.predict(future)

def create_candlestick_chart(forecast):
    fig = go.Figure(data=[go.Candlestick(
        x=forecast['ds'],
        open=forecast['yhat_lower'],
        high=forecast['yhat'],
        low=forecast['yhat_lower'],
        close=forecast['yhat_upper']
    )])
    
    fig.update_layout(
        title='Demand Forecasting Candlestick',
        xaxis_title='Date',
        yaxis_title='Forecast',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2,
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            gridwidth=2,
        ),
    )

    return pio.to_html(fig, full_html=False)

def get_start_date():
    today = datetime.today()
    start_date = today - timedelta(days=today.weekday())
    start_date = start_date.strftime('%Y-%m-%d')
    return start_date

@app.route('/', methods=['GET', 'POST'])
def index():
    chart = None
    if request.method == 'POST':
        periods = int(request.form.get('periods', 52))  # Default to 52 weeks if not provided
        start_date = get_start_date()
        future = create_future_dataframe(start_date, periods)
        forecast = make_forecast(model, future)
        chart = create_candlestick_chart(forecast)
    return render_template('index.html', chart=chart)

if __name__ == '__main__':
    app.run(debug=True)
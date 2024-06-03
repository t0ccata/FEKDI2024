from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = Flask(__name__)

# Load the model once when the app starts
model = joblib.load('weekly_prophet_model.pkl')

def predict_future(model, start_date, periods):
    future = pd.date_range(start=start_date, periods=periods, freq='W-SUN').to_frame(index=False, name='ds')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    periods = data.get('periods', 52)  # Default to 52 weeks if not provided
    start_date = '2024-06-03'
    forecast = predict_future(model, start_date, periods)
    return jsonify(forecast.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

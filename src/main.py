from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to prepare the data for a specific day over the years
def prepare_training_data(df, target_date):
    month = target_date.month
    day = target_date.day
    filtered_data = df[(df['Month'] == month) & (df['Day'] == day)]
    filtered_data = filtered_data.sort_values(by='Year')
    return filtered_data[['Load']].values

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future load for a given date
def predict_future_load(df, target_date, epochs=100, batch_size=32):
    training_data = prepare_training_data(df, target_date)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_data = scaler.fit_transform(training_data)

    X_train, y_train = [], []
    for i in range(3, len(scaled_training_data)):
        X_train.append(scaled_training_data[i-3:i, 0])
        y_train.append(scaled_training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    last_3_years_load = scaled_training_data[-3:].reshape(1, 3, 1)
    predicted_load = model.predict(last_3_years_load)
    predicted_load = scaler.inverse_transform(predicted_load)
    return predicted_load[0][0]

# Function to predict a series of future loads for multiple days
def predict_load_series(df, start_date: pd.Timestamp, period_days: int):
    predicted_loads = []
    actual_loads = []

    for i in range(period_days):
        target_date = start_date + pd.Timedelta(days=i)
        try:
            load = predict_future_load(df, target_date)
            predicted_loads.append(round(float(load), 2))
            actual_loads.append(None)
        except Exception as e:
            logging.error(f"Error predicting for {target_date}: {str(e)}")
            predicted_loads.append(None)
            actual_loads.append(None)

    mape = None
    return {
        "predicted_load": predicted_loads,
        "actual_load": actual_loads,
        "mape": mape
    }

# Load and preprocess CSV once to avoid repeated I/O
def load_and_prepare_csv():
    df = pd.read_csv("delhi_load_with_peaks - delhi_load_with_peaks_modi.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    return df

@app.get("/")
async def root():
    return {"message": "Welcome to Power Demand Prediction API"}

@app.get("/predict/day")
async def predict_day(date: str = Query(..., description="YYYY-MM-DD format")):
    try:
        custom_date = pd.to_datetime(date)
        df = load_and_prepare_csv()
        result = predict_load_series(df, custom_date, 1)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/week")
async def predict_week(date: str = Query(...)):
    try:
        custom_date = pd.to_datetime(date)
        df = load_and_prepare_csv()
        result = predict_load_series(df, custom_date, 7)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/month")
async def predict_month(date: str = Query(...)):
    try:
        custom_date = pd.to_datetime(date)
        df = load_and_prepare_csv()
        result = predict_load_series(df, custom_date, 30)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

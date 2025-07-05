import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# Load the data
df = pd.read_csv('delhi_load_with_peaks - delhi_load_with_peaks_modi.csv')

# Convert Datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract month and year from the Datetime
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day

# Function to prepare the data for a specific day over the years
def prepare_training_data(df, target_date):
    month = target_date.month
    day = target_date.day
    
    # Filter data for the same day across different years
    filtered_data = df[(df['Month'] == month) & (df['Day'] == day)]
    
    # Sort by year
    filtered_data = filtered_data.sort_values(by='Year')
    
    return filtered_data[['Load']].values

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Output a single prediction (Load)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future load
def predict_future_load(df, target_date, epochs=100, batch_size=32):
    # Prepare training data
    training_data = prepare_training_data(df, target_date)

    # Scale the data (normalization)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_data = scaler.fit_transform(training_data)
    
    # Prepare the input for the LSTM model
    X_train = []
    y_train = []
    
    # Use a time window of 3 years
    for i in range(3, len(scaled_training_data)):
        X_train.append(scaled_training_data[i-3:i, 0])
        y_train.append(scaled_training_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape the data to be suitable for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Create LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Predict the load for the given target date
    last_3_years_load = scaled_training_data[-3:].reshape(1, 3, 1)
    predicted_load = model.predict(last_3_years_load)
    
    # Inverse scale to get the actual load value
    predicted_load = scaler.inverse_transform(predicted_load)
    
    return predicted_load[0][0]

# Define the target date for which to predict load (YYYY-MM-DD format)
target_date = datetime(2024, 9, 20)

# Run the prediction
predicted_load = predict_future_load(df, target_date)
print(f'Predicted Load for {target_date.date()}: {predicted_load}')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime

# Function to create sequences from the dataset
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Function to plot actual vs predicted values
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual Load", color='blue')
    plt.plot(predicted, label="Predicted Load", color='orange')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    plt.show()

# LSTM prediction function with plotting
def lstm_train_and_predict(df, date_to_predict, column_name='Load', time_steps=24):
    # Convert the Datetime column to pandas datetime object
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Filter data for the same month over previous years
    month_to_predict = date_to_predict.month
    past_data = df[df.index.month == month_to_predict][column_name].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_data_scaled = scaler.fit_transform(past_data.reshape(-1, 1))
    
    # Prepare training data
    X, y = create_sequences(past_data_scaled, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, batch_size=1, epochs=3)
    model.save("lstm_model.keras")

    # Get the past data to make predictions
    test_input = past_data_scaled[-time_steps:].reshape(1, time_steps, 1)

    # Make the prediction
    predicted_scaled = model.predict(test_input)
    predicted_value = scaler.inverse_transform(predicted_scaled)
    
    # Plot actual vs predicted
    plot_predictions(past_data[-100:], np.concatenate((past_data[-99:], predicted_value.flatten()), axis=0), "Actual vs Predicted Load")


    return predicted_value[0][0]


predicted_value = lstm_train_and_predict(df, datetime(2024, 8 , 21))
print(f"Predicted Load for 2024-09-20: {predicted_value}")

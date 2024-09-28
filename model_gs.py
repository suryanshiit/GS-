import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def preprocess_data(df, symbol_id, past_days=50):
    """
    Preprocess data for a single symbol_id.
    This function prepares the input data for the LSTM Model 
    """
    symbol_data = df[df['symbol_id'] == symbol_id].copy()

    symbol_data.sort_values(by=['date_id', 'seconds_bucket'], inplace=True)

    # To extract cumulative continuous volume data for all time buckets (except 31,200 seconds)
    cumulative_volumes = symbol_data[symbol_data['seconds_bucket'] < 31200]

    pivoted_data = cumulative_volumes.pivot(index='date_id', columns='seconds_bucket', values='cummulative_continous_volume')

    # handling nan values
    pivoted_data.fillna(0, inplace=True)

    # extracting the closing_volume data
    target_data = symbol_data[symbol_data['seconds_bucket'] == 31200][['date_id', 'close_volume']].set_index('date_id')

    # Checking available days for training
    available_days = min(past_days, len(pivoted_data))

    # checking that we only take available_days for training
    pivoted_data_train = pivoted_data.iloc[:available_days]
    target_data_train = target_data.iloc[:available_days]

    X = []
    y = []
    for i in range(available_days):
        # Cumulative volume data 
        X.append(pivoted_data_train.values[i])
        # close_volume data
        y.append(target_data_train['close_volume'])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    #print(f"Training data shape (X): {X.shape}")
    #print(f"Training data shape (y): {y.shape}")
    return X, y


def create_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(500, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) 

    model.add(LSTM(300, activation='relu', return_sequences=False))
    model.add(Dropout(0.2)) 

    model.add(Dense(128, activation='relu')) 
    model.add(Dropout(0.2)) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))  
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2)) 
   
    model.add(Dense(1))  
    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model

def train_and_predict(df):
    """
    Main function to train an LSTM model for each symbol_id and predict the close_volume for day 51.
    """
    # Get all unique symbol_ids
    symbol_ids = df['symbol_id'].unique()

    # Prepare to store results
    results = []

    # Iterate over each symbol_id
    for symbol_id in symbol_ids:
        print(f"Processing symbol_id: {symbol_id}")

        # Preprocess data for this symbol_id (get past 50 days' cumulative volume)
        X, y = preprocess_data(df, symbol_id)

        # Split into training and testing sets
        X_train, X_test = X[:-1], X[-1:]  # The last sample is for day 51
        y_train = y[:-1]  # The target close_volume values for training
        print(y_train)
        print(X_train)
        # Create and train the LSTM model
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        # Predict the close_volume for day 51
        y_pred = model.predict(X_test)[0][0]

        # Store the result for this symbol_id
        results.append([symbol_id, y_pred])

    # Prepare the final output
    results_df = pd.DataFrame(results, columns=['symbol_id', 'close_volume'])

    # Print the result in the expected format
    print("symbol_id,close_volume")
    for index, row in results_df.iterrows():
        print(f"{int(row['symbol_id'])},{row['close_volume']:.1f}")

# Load the dataset
filename = '/content/sample1.csv'
df = pd.read_csv(filename)

# Run the training and prediction pipeline
train_and_predict(df)

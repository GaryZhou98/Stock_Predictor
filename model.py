import argparse
import pandas as pd


from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,LSTM, Dropout, BatchNormalization
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import ipdb

#preprocessing constants
TIME_STEP = 3
PREDICTION_STEP = 1

#model constants
LSTM_OUTPUT_SIZE = 256
DENSE1_OUTPUT_SIZE = 64
DENSE2_OUTPUT_SIZE = 32
DENSE3_OUTPUT_SIZE = 1



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    return parser.parse_args()


def series_to_sl(values):
    # values = scaler.transform(values)

    #Split data into input and labels
    input_data = np.array(values[0: len(values)-PREDICTION_STEP])
    label = np.array(values[TIME_STEP:, 0]).reshape(-1,1)

    #Normalization of data
    for i in range(0, input_data.shape[1]):
        temp = input_data[:, i].reshape(-1,1)
        input_data[:, i] = MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]

    # x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(input_data)
    # input_data = x_scaler.transform(input_data)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(label)
    label = y_scaler.transform(label)

    x = []
    y = []

    #reshape input and label
    for start, end in zip(range(0, len(input_data) - TIME_STEP), range(TIME_STEP, len(input_data))):
        x.append(input_data[start:end])
    for start, end in zip(range(0, len(label) - PREDICTION_STEP), range(PREDICTION_STEP, len(label))):
        y.append(label[start:end][0])
    
    x = np.array(x)
    y = np.array(y)

    #ipdb.set_trace()
    return x, y, y_scaler


def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, header=0, index_col=0)
    values = data.to_numpy()
    return values.astype("float32")


def prepare_data(filepath):
    data_values = load_data_from_csv(filepath)
    input_data, label, y_scaler = series_to_sl(data_values)
    x_train, x_test, y_train, y_test = train_test_split(input_data, label, test_size = 0.25)
    #ipdb.set_trace()
    return x_train, x_test, y_train, y_test, y_scaler


def build_model(x_train, y_train, x_test, y_test):
  model = Sequential()
  model.add(LSTM(LSTM_OUTPUT_SIZE, input_shape=x_train.shape[1:], return_sequences=True))
  model.add(Dropout(0.1))
  model.add(BatchNormalization())

  model.add(LSTM(LSTM_OUTPUT_SIZE, return_sequences=False))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())


  model.add(Dense(DENSE1_OUTPUT_SIZE, activation='relu'))
  model.add(Dense(DENSE2_OUTPUT_SIZE, activation='relu'))
  model.add(Dense(DENSE3_OUTPUT_SIZE, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam')

  model.fit(x_train, y_train)
  score = model.evaluate(x_test, y_test)
  print("Validation accuracy percentage", score*100)

  return model

def model_prediction(symbol):
  x_train, x_test, y_train, y_test, scaler = prepare_data(f"{symbol}_daily.csv")
  
  model = build_model(x_train, y_train, x_test, y_test)
  prediction = model.predict(x_test)
  prediction = scaler.inverse_transform(prediction)
  pyplot.plot(prediction, label='prediction')
  pyplot.plot(scaler.inverse_transform(y_test), label='actual')
  pyplot.legend()
  pyplot.show()


if __name__ == "__main__":
    args = get_args()
    if args.symbol:
        model_prediction(args.symbol)
        exit(1)
    raise ValueError(
        "Missing symbol"
    )
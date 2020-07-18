import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

TIME_STEP = 3
PREDICTION_STEP = 1

def series_to_sl(values):
    #Normalization of data
    values = MinMaxScaler(feature_range=(0, 1)).fit_transform(values)

    #Split data into input and labels
    input_data = values[0: len(values)-PREDICTION_STEP]
    label = values[TIME_STEP:, 0]
    x = []
    y = []

    #reshape input and label
    for start, end in zip(range(0, len(input_data) - TIME_STEP), range(TIME_STEP, len(input_data))):
        x.append(input_data[start:end])
    for start, end in zip(range(0, len(label) - PREDICTION_STEP), range(PREDICTION_STEP, len(label))):
        y.append(label[start:end])
    return np.array(x), np.array(y)


def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, header=0, index_col=0)
    values = data.to_numpy()
    return values.astype("float32")


def prepare_data(filepath):
    data_values = load_data_from_csv(filepath)
    input_data, label = series_to_sl(data_values)
    x_train, x_test, y_train, y_test = train_test_split(input_data, label, test_size = 0.25)
    return x_train, x_test, y_train, y_test
prepare_data("FB_daily.csv")

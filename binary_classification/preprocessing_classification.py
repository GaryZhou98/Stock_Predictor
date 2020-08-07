import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import constants 


def series_to_sl(values):
    # Split data into input and labels
    input_data = np.array(values[:len(values) - constants.PREDICTION_STEP])
    for i in range(0, len(input_data)):
      input_data[i][0] = 1 if values[i + 1][1] > values[i][1] else 0
    
    # Normalization of input data
    for i in range(1, input_data.shape[1]):
        temp = input_data[:, i].reshape(-1, 1)
        input_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )
    
    x = input_data[:, 1:]
    y = input_data[:, 0]

    return x, y


def prepare_data(filepath, shuffle=False):
    price_df = pd.read_csv(filepath, header=0)
    price_df.drop(["date"], axis=1, inplace=True)
    price_df.insert(0, "label", 0)
    price_values = price_df.to_numpy().astype("float32")

    input_data, label, = series_to_sl(price_values)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, label, test_size=constants.TEST_PORTION, shuffle=shuffle
    )

    price_data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return price_data


def prepare_prediction_data(filepath):
    price_df = pd.read_csv(filepath, header=0)
    price_df.drop(["date"], axis=1, inplace=True)
    price_values = price_df.to_numpy().astype("float32")

    price_data = np.array(price_values[len(price_values) - c.PREDICTION_STEP :])
    for i in range(0, price_data.shape[1]):
        temp = price_data[:, i].reshape(-1, 1)
        price_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )
    price_predic_data = {"x_test": price_x}

    return price_predic_data
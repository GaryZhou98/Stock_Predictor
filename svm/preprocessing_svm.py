import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ipdb
import sys
sys.path.append('../')

import constants


def regression_series_to_sl(values, is_covid=False):
    # Split data into input and labels
    shift_period = -1 * constants.PREDICTION_STEP
    label = values[:shift_period, 0]
    input_data = np.array(values[0 : shift_period, 1:])

    # Normalization of data
    for i in range(0, input_data.shape[1]):
        temp = input_data[:, i].reshape(-1, 1)
        input_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )

    label = label.reshape(-1, 1)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(label)
    label = y_scaler.transform(label)
    # ipdb.set_trace()
    return input_data, label[:,0], y_scaler


def prepare_regression_data(filepath, shuffle=True):
    price_df = pd.read_csv(filepath, header=0)
    price_df.drop(["date"], axis=1, inplace=True)
    price_df.insert(0, "label", 0)
    price_df['label'] = price_df[['close']].shift(-1 * constants.PREDICTION_STEP)
    price_values = price_df.to_numpy().astype("float32")

    input_data, label, scaler = regression_series_to_sl(price_values, is_covid=False)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, label, test_size=constants.TEST_PORTION, shuffle=shuffle
    )

    price_data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }

    return price_data

def binary_series_to_sl(values):
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


def prepare_binary_data(filepath, shuffle=False):
    price_df = pd.read_csv(filepath, header=0)
    price_df.drop(["date"], axis=1, inplace=True)
    price_df.insert(0, "label", 0)
    price_values = price_df.to_numpy().astype("float32")

    input_data, label, = binary_series_to_sl(price_values)
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
    covid_df = pd.read_csv(constants.SUBDIR_COVID_PATH, header=0)
    covid_df = price_df.merge(covid_df, how="left", on="date")
    covid_df.fillna(value=0, inplace=True)
    covid_df = covid_df[["close"] + c.COVID_COLUMNS]
    covid_df.drop(["date"], axis=1, inplace=True)
    price_df.drop(["date"], axis=1, inplace=True)
    price_values = price_df.to_numpy().astype("float32")
    covid_values = covid_df.to_numpy().astype("float32")

    covid_data = np.array(covid_values[len(covid_values) - c.PREDICTION_STEP :])
    for i in range(0, covid_data.shape[1]):
        temp = covid_data[:, i].reshape(-1, 1)
        covid_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )
    covid_data = covid_data[:, 1:]

    price_data = np.array(price_values[len(price_values) - c.PREDICTION_STEP :])
    for i in range(0, price_data.shape[1]):
        temp = price_data[:, i].reshape(-1, 1)
        price_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )

    covid_x = covid_data.reshape(1, covid_data.shape[0], covid_data.shape[1])
    price_x = price_data.reshape(1, price_data.shape[0], price_data.shape[1])

    price_predic_data = {"x_test": price_x}
    covid_predic_data = {"x_test": covid_x}
    return price_predic_data, covid_predic_data

if __name__ == "__main__":
    prepare_serial_data(constants.SUBDIR_CSV_PATH.format('CCL'), shuffle=constants.SAMPLE_SHUFFLE)

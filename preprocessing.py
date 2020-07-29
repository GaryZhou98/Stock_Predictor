import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import constants


def series_to_sl(values, is_covid=False):
    # Split data into input and labels
    input_data = np.array(values[0: len(values) - constants.PREDICTION_STEP])
    label = np.array(values[constants.TIME_STEP:, 0]).reshape(-1, 1)

    # Normalization of data
    for i in range(0, input_data.shape[1]):
        temp = input_data[:, i].reshape(-1, 1)
        input_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )

    # x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(input_data)
    # input_data = x_scaler.transform(input_data)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(label)
    label = y_scaler.transform(label)

    x = []
    y = []

    # reshape input and label
    for start, end in zip(
        range(0, len(input_data) - constants.TIME_STEP),
        range(constants.TIME_STEP, len(input_data)),
    ):
        if is_covid:
            x.append(input_data[start:end, 1:])
        else:
            x.append(input_data[start:end])
    for start, end in zip(
        range(0, len(label) - constants.PREDICTION_STEP),
        range(constants.PREDICTION_STEP, len(label)),
    ):
        y.append(label[start:end, 0])

    x = np.array(x)
    y = np.array(y)
    return x, y, y_scaler


def prepare_data(filepath, shuffle=True):

    price_df = pd.read_csv(filepath, header=0)
    covid_df = pd.read_csv("covid.csv", header=0)
    covid_df = price_df.merge(covid_df, how="left", on="date")
    covid_df.fillna(value=0, inplace=True)
    covid_df = covid_df[["close"] + constants.COVID_COLUMNS]
    covid_df.drop(["date"], axis=1, inplace=True)
    price_df.drop(["date"], axis=1, inplace=True)
    price_values = price_df.to_numpy().astype("float32")
    covid_values = covid_df.to_numpy().astype("float32")

    covid_input, covid_label, scaler = series_to_sl(
        covid_values, is_covid=True)

    test_portion = int(covid_label.shape[0] * 0.15)
    overall_portion = int(covid_label.shape[0] * 0.25)
    x_train, x_test, y_train, y_test = train_test_split(
        covid_input, covid_label, test_size=test_portion, shuffle=shuffle
    )

    # Further split training data to train for individual model vs. train for overall model

    x_train_overall = x_train[len(x_train) - overall_portion:, :, :]
    y_train_overall = y_train[len(y_train) - overall_portion:, :]
    x_train_individual = x_train[: len(x_train) - overall_portion, :, :]
    y_train_individual = y_train[: len(y_train) - overall_portion, :]

    covid_data = {
        "x_train_individual": x_train_individual,
        "x_test": x_test,
        "y_train_individual": y_train_individual,
        "y_test": y_test,
        "x_train_overall": x_train_overall,
        "y_train_overall": y_train_overall,
        "scaler": scaler,
    }

    input_data, label, scaler = series_to_sl(price_values, is_covid=False)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, label, test_size=test_portion, shuffle=shuffle
    )

    x_train_overall = x_train[len(x_train) - overall_portion:, :, :]
    y_train_overall = y_train[len(y_train) - overall_portion:, :]
    x_train_individual = x_train[: len(x_train) - overall_portion, :, :]
    y_train_individual = y_train[: len(y_train) - overall_portion, :]

    price_data = {
        "x_train_individual": x_train_individual,
        "x_test": x_test,
        "y_train_individual": y_train_individual,
        "y_test": y_test,
        "x_train_overall": x_train_overall,
        "y_train_overall": y_train_overall,
        "scaler": scaler,
    }

    return price_data, covid_data


def prepare_prediction_data(filepath):
    price_df = pd.read_csv(filepath, header=0)
    covid_df = pd.read_csv("covid.csv", header=0)
    covid_df = price_df.merge(covid_df, how="left", on="date")
    covid_df.fillna(value=0, inplace=True)
    covid_df = covid_df[["close"] + c.COVID_COLUMNS]
    covid_df.drop(["date"], axis=1, inplace=True)
    price_df.drop(["date"], axis=1, inplace=True)
    price_values = price_df.to_numpy().astype("float32")
    covid_values = covid_df.to_numpy().astype("float32")

    covid_data = np.array(covid_values[len(covid_values) - c.PREDICTION_STEP:])
    for i in range(0, covid_data.shape[1]):
        temp = covid_data[:, i].reshape(-1, 1)
        covid_data[:, i] = (
            MinMaxScaler(feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]
        )
    covid_data = covid_data[:, 1:]

    price_data = np.array(price_values[len(price_values) - c.PREDICTION_STEP:])
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

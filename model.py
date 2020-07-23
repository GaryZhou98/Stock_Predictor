import argparse
import pandas as pd


from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Input, Activation, LSTM, Dropout, BatchNormalization, concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import ipdb

# preprocessing constants
TIME_STEP = 5
PREDICTION_STEP = 1

# model constants
LSTM_OUTPUT_SIZE = 128
DENSE1_OUTPUT_SIZE = 64
DENSE2_OUTPUT_SIZE = 32
DENSE3_OUTPUT_SIZE = PREDICTION_STEP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--symbol", help="symbol to pull data on", default=False)
    return parser.parse_args()


def series_to_sl(values, is_covid=False):
    # values = scaler.transform(values)

    # Split data into input and labels
    input_data = np.array(values[0: len(values)-PREDICTION_STEP])
    label = np.array(values[TIME_STEP:, 0]).reshape(-1, 1)

    # Normalization of data
    for i in range(0, input_data.shape[1]):
        temp = input_data[:, i].reshape(-1, 1)
        input_data[:, i] = MinMaxScaler(
            feature_range=(0, 1)).fit(temp).transform(temp)[:, 0]

    # x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(input_data)
    # input_data = x_scaler.transform(input_data)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(label)
    label = y_scaler.transform(label)

    x = []
    y = []

    # reshape input and label

    for start, end in zip(range(0, len(input_data) - TIME_STEP), range(TIME_STEP, len(input_data))):
        if is_covid:
            x.append(input_data[start:end, 1:])
        else:
            x.append(input_data[start:end])
    for start, end in zip(range(0, len(label) - PREDICTION_STEP), range(PREDICTION_STEP, len(label))):
        y.append(label[start:end, 0])

    x = np.array(x)
    y = np.array(y)
    # ipdb.set_trace()
    return x, y, y_scaler


def prepare_data(filepath, shuffle=True):

    price_df = pd.read_csv(filepath, header=0)
    covid_df = pd.read_csv('covid.csv', header=0)
    covid_df = price_df.merge(covid_df, how='left', on='date')
    covid_df.fillna(value=0, inplace=True)
    covid_df = covid_df[['close', 'date', 'cases', 'deaths']]
    covid_df.drop(['date'], axis=1, inplace=True)
    price_df.drop(['date'], axis=1, inplace=True)
    price_values = price_df.to_numpy().astype("float32")
    covid_values = covid_df.to_numpy().astype("float32")

    covid_input, covid_label, scaler = series_to_sl(
        covid_values, is_covid=True)

    test_portion = int(covid_label.shape[0]*0.15)
    overall_portion = int(covid_label.shape[0]*0.25)
    x_train, x_test, y_train, y_test = train_test_split(
        covid_input, covid_label, test_size=test_portion, shuffle=shuffle)

    # Further split training data to train for individual model vs. train for overall model

    x_train_overall = x_train[len(x_train)-overall_portion:, :, :]
    y_train_overall = y_train[len(y_train)-overall_portion:, :]
    x_train_individual = x_train[:len(x_train)-overall_portion, :, :]
    y_train_individual = y_train[:len(y_train)-overall_portion, :]

    covid_data = {"x_train_individual": x_train_individual,
                  "x_test": x_test,
                  "y_train_individual": y_train_individual,
                  "y_test": y_test,
                  "x_train_overall": x_train_overall,
                  "y_train_overall": y_train_overall,
                  "scaler": scaler}

    input_data, label, scaler = series_to_sl(price_values, is_covid=False)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, label, test_size=test_portion, shuffle=shuffle)

    x_train_overall = x_train[len(x_train)-overall_portion:, :, :]
    y_train_overall = y_train[len(y_train)-overall_portion:, :]
    x_train_individual = x_train[:len(x_train)-overall_portion, :, :]
    y_train_individual = y_train[:len(y_train)-overall_portion, :]

    price_data = {"x_train_individual": x_train_individual,
                  "x_test": x_test,
                  "y_train_individual": y_train_individual,
                  "y_test": y_test,
                  "x_train_overall": x_train_overall,
                  "y_train_overall": y_train_overall,
                  "scaler": scaler}

    return price_data, covid_data


def build_covid_model(x_train, y_train, batch_size, epochs,):
    covid_model = Sequential()
    covid_data = x_train
    covid_model.add(LSTM(LSTM_OUTPUT_SIZE, input_shape=(
        covid_data.shape[1:]), return_sequences=True))
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(LSTM(LSTM_OUTPUT_SIZE, return_sequences=False))
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(Dense(DENSE2_OUTPUT_SIZE, activation='relu'))
    covid_model.add(Dense(DENSE3_OUTPUT_SIZE, activation='sigmoid'))

    covid_model.compile(loss='mse', optimizer='adam')

    covid_model.fit(covid_data, y_train, batch_size=batch_size,
                    epochs=epochs, shuffle=True)
    return covid_model


def build_price_model(x_train, y_train, batch_size, epochs,):
    price_model = Sequential()
    price_data = x_train

    price_model.add(LSTM(LSTM_OUTPUT_SIZE, input_shape=(
        price_data.shape[1:]), return_sequences=False))
    price_model.add(Dropout(0.2))
    price_model.add(BatchNormalization())

    # price_model.add(LSTM(LSTM_OUTPUT_SIZE,return_sequences=False))
    # price_model.add(Dropout(0.2))
    # price_model.add(BatchNormalization())

    price_model.add(Dense(DENSE2_OUTPUT_SIZE, activation='relu'))
    price_model.add(Dense(DENSE3_OUTPUT_SIZE, activation='sigmoid'))

    price_model.compile(loss='mse', optimizer='adam')

    price_model.fit(price_data, y_train, batch_size=batch_size,
                    epochs=epochs, shuffle=True)
    return price_model


def build_overall_model(price_data, covid_data, batch_size, epochs,):
    # covid_model

    input_models = [
        build_covid_model(covid_data['x_train_individual'],
                          covid_data['y_train_individual'], batch_size, epochs//100),
        build_price_model(price_data['x_train_individual'],
                          price_data['y_train_individual'], batch_size, epochs)
    ]
    for i in range(len(input_models)):
        model = input_models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name


    # define multi-headed input
    ensemble_inputs = [model.input for model in input_models]
    # ipdb.set_trace()
	# concatenate merge output from each model
    ensemble_outputs = [model.output for model in input_models]
    merge = concatenate(ensemble_outputs)

    hidden1 = Dense(DENSE1_OUTPUT_SIZE, activation='relu')(merge)
    dropout1 = Dropout(0.2)(hidden1)
    batch_norm1 = BatchNormalization()(dropout1)
    hidden2 = Dense(DENSE2_OUTPUT_SIZE, activation='relu')(batch_norm1)
    dropout2 = Dropout(0.2)(hidden2)
    batch_norm2 = BatchNormalization()(dropout2)
    output = Dense(DENSE3_OUTPUT_SIZE, activation='sigmoid')(batch_norm2)
    model = Model(inputs=ensemble_inputs, outputs=output)

    model.compile(loss="mse", optimizer="adam")


    return model


def fit_overall_model(model, price_data, covid_data, epochs):
    covid_input = covid_data['x_train_overall']
    price_input = price_data['x_train_overall']
    covid_label = covid_data['y_train_overall']
    price_label = price_data['y_train_overall']
    # ipdb.set_trace()
# fit model
    model.fit([covid_input, price_input], covid_label,
              epochs=epochs, shuffle=True)

# make a prediction with a stacked model


def predict_overall_model(model, price_data, covid_data,):
        # prepare input data
    covid_input = covid_data['x_test']
    price_input = price_data['x_test']
# make prediction
    return model.predict([covid_input, price_input], verbose=0)


def model_prediction(symbol):
    prediction = []

    price_data, covid_data = prepare_data(
        f"{symbol}_daily.csv", shuffle=False)

    model = build_overall_model(
        price_data, covid_data, batch_size=10, epochs=200)
    fit_overall_model(model, price_data, covid_data, epochs=200)

    prediction = predict_overall_model(model, price_data, covid_data)
    actual_full = np.append(
        price_data['y_train_individual'], price_data['y_train_overall'])
    actual_full = np.append(actual_full, price_data['y_test'])
    prediction_full = np.append(
        covid_data['y_train_individual'], covid_data['y_train_overall'])
    prediction_full = np.append(prediction_full, prediction)

    ipdb.set_trace()
    prediction = covid_data['scaler'].inverse_transform(
        prediction_full.reshape(-1, 1))
    actual = price_data['scaler'].inverse_transform(actual_full.reshape(-1, 1))
    dates = pd.read_csv(f"{symbol}_daily.csv", header=0)['date'].values
    dates = dates[TIME_STEP:, ]
    dates = dates[:-PREDICTION_STEP, ]
    pyplot.plot(dates, prediction, label='prediction')
    pyplot.plot(dates, actual, label='actual')    
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

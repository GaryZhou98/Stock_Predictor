import argparse
import constants as c
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    concatenate,
)
from tensorflow.keras.models import Model, load_model
from preprocessing import prepare_data, prepare_prediction_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    return parser.parse_args()


def build_covid_model(x_train, y_train, batch_size):
    covid_model = Sequential()
    covid_data = x_train
    covid_model.add(
        LSTM(
            c.LSTM_OUTPUT_SIZE,
            input_shape=(covid_data.shape[1:]),
            return_sequences=True,
        )
    )
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(LSTM(c.LSTM_OUTPUT_SIZE, return_sequences=False))
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(Dense(c.DENSE1_OUTPUT_SIZE, activation="relu"))
    covid_model.add(Dense(c.DENSE2_OUTPUT_SIZE, activation="relu"))
    covid_model.add(Dense(c.DENSE3_OUTPUT_SIZE, activation="sigmoid"))

    covid_model.compile(loss="mse", optimizer="adam")

    covid_model.fit(
        covid_data,
        y_train,
        batch_size=batch_size,
        epochs=c.COVID_EPOCH,
        shuffle=c.TRAIN_SHUFFLE,
    )
    return covid_model


def build_price_model(x_train, y_train, batch_size):
    price_model = Sequential()
    price_data = x_train

    price_model.add(
        LSTM(
            c.LSTM_OUTPUT_SIZE,
            input_shape=(price_data.shape[1:]),
            return_sequences=False,
        )
    )
    price_model.add(Dropout(0.2))
    price_model.add(BatchNormalization())

    price_model.add(Dense(c.DENSE1_OUTPUT_SIZE, activation="relu"))
    price_model.add(Dense(c.DENSE2_OUTPUT_SIZE, activation="relu"))
    price_model.add(Dense(c.DENSE3_OUTPUT_SIZE, activation="sigmoid"))

    price_model.compile(loss="mse", optimizer="adam")

    price_model.fit(
        price_data,
        y_train,
        batch_size=batch_size,
        epochs=c.PRICE_EPOCH,
        shuffle=c.TRAIN_SHUFFLE,
    )
    return price_model


def build_overall_model(price_data, covid_data, batch_size):
    input_models = [
        build_covid_model(
            covid_data["x_train_individual"],
            covid_data["y_train_individual"],
            batch_size,
        ),
        build_price_model(
            price_data["x_train_individual"],
            price_data["y_train_individual"],
            batch_size,
        ),
    ]
    for i in range(len(input_models)):
        model = input_models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = "ensemble_" + str(i + 1) + "_" + layer.name

    # define multi-headed input
    ensemble_inputs = [model.input for model in input_models]

    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in input_models]
    merge = concatenate(ensemble_outputs)

    hidden1 = Dense(c.DENSE1_OUTPUT_SIZE, activation="relu")(merge)
    dropout1 = Dropout(0.2)(hidden1)
    batch_norm1 = BatchNormalization()(dropout1)
    hidden2 = Dense(c.DENSE2_OUTPUT_SIZE, activation="relu")(batch_norm1)
    dropout2 = Dropout(0.2)(hidden2)
    batch_norm2 = BatchNormalization()(dropout2)
    output = Dense(c.DENSE3_OUTPUT_SIZE, activation="sigmoid")(batch_norm2)
    model = Model(inputs=ensemble_inputs, outputs=output)

    model.compile(loss="mse", optimizer="adam")

    return model


def fit_overall_model(model, price_data, covid_data):
    covid_input = covid_data["x_train_overall"]
    price_input = price_data["x_train_overall"]
    covid_label = covid_data["y_train_overall"]

    # fit model
    model.fit(
        [covid_input, price_input],
        covid_label,
        epochs=c.OVERALL_EPOCH,
        shuffle=c.TRAIN_SHUFFLE,
    )


# make a prediction with a stacked model
def predict_overall_model(model, price_data, covid_data):
    # prepare input data
    covid_input = covid_data["x_test"]
    price_input = price_data["x_test"]
    # make prediction
    return model.predict([covid_input, price_input], verbose=2)


def model_prediction(symbol):
    prediction = []
    scaler = None
    for i in range(0, c.NUM_TRAIN):
        price_data, covid_data = prepare_data(
            c.SAVED_CSV_PATH.format(symbol), shuffle=c.SAMPLE_SHUFFLE
        )
        scaler = covid_data["scaler"]
        print(f"Executing for the {i}th time")
        model = build_overall_model(price_data, covid_data, batch_size=10)
        fit_overall_model(model, price_data, covid_data)

        price_data, covid_data = prepare_data(
            c.SAVED_CSV_PATH.format(symbol), shuffle=False
        )

        print(f"Making {i}th Prediction")

        if i == 0:
            prediction = predict_overall_model(model, price_data, covid_data)
        else:
            prediction += predict_overall_model(model, price_data, covid_data)
    prediction = np.array(prediction / c.NUM_TRAIN)
    label = np.array(covid_data["y_test"][:, 2]).reshape(-1, 1)
    pyplot.plot(
        scaler.inverse_transform(prediction[:, 0].reshape(-1, 1)),
        label="prediction_first",
    )
    pyplot.plot(
        scaler.inverse_transform(prediction[:, c.PREDICTION_STEP - 1].reshape(-1, 1)),
        label="prediction_last",
    )
    pyplot.plot(scaler.inverse_transform(label), label="actual")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    args = get_args()
    if args.symbol:
        model_prediction(args.symbol)
        exit(1)
    raise ValueError("Missing symbol")

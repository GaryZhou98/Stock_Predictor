import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import constants as c
from matplotlib import pyplot
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    concatenate,
)
from binary_classification.preprocessing_classification import prepare_data, prepare_prediction_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    return parser.parse_args()

def build_model(price_data, batch_size):
    x_train = price_data['x_train']
    y_train = price_data['y_train']
    price_model = Sequential()
    price_model.add(Dense(c.DENSE1_OUTPUT_SIZE, input_shape=x_train.shape[1:], activation="relu"))
    price_model.add(Dropout(0.2))
    price_model.add(BatchNormalization())
    price_model.add(Dense(c.DENSE2_OUTPUT_SIZE, activation="relu"))
    price_model.add(Dropout(0.2))
    price_model.add(BatchNormalization())
    price_model.add(Dense(c.CLASSFICATION_OUTPUT_SIZE, activation="softmax"))

    price_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    price_model.fit(x_train, y_train, epochs=c.CLASSFICATION_EPOCH, batch_size=batch_size, shuffle=False)
    return price_model

def model_prediction(symbol):
    price_data = prepare_data(c.SUBDIR_CSV_PATH.format(symbol), shuffle=c.SAMPLE_SHUFFLE)
    model = build_model(price_data, 20)

    test_loss, test_acc = model.evaluate(price_data['x_test'], price_data['y_test'])
    print('\nloss: ', test_loss)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    args = get_args()
    if args.symbol:
        model_prediction(args.symbol)
        exit(1)
    raise ValueError("Missing symbol")
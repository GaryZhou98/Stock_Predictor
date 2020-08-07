import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import constants as c
from matplotlib import pyplot
from sklearn import svm
from svm.preprocessing_svm import prepare_regression_data, prepare_binary_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    return parser.parse_args()

def build_binary_model(price_data, batch_size):
    x_train = price_data['x_train']
    y_train = price_data['y_train']
    
    price_model = svm.SVC(C=1, kernel='linear', gamma=0.01)
    price_model.fit(x_train, y_train)
    return price_model

def build_regression_model(price_data, batch_size):
    x_train = price_data['x_train']
    y_train = price_data['y_train']
    
    price_model = svm.SVR(C=1, kernel='rbf', gamma=0.1)
    price_model.fit(x_train, y_train)

    return price_model

def model_prediction(symbol):
    regression_price_data = prepare_regression_data(c.SUBDIR_CSV_PATH.format(symbol), shuffle=c.SAMPLE_SHUFFLE)
    regression_model = build_regression_model(regression_price_data, 20)
    regression_test_score = regression_model.score(regression_price_data['x_test'], regression_price_data['y_test'])
    print('\nRegression Test accuracy:', regression_test_score)

    binary_price_data = prepare_binary_data(c.SUBDIR_CSV_PATH.format(symbol), shuffle=c.SAMPLE_SHUFFLE)
    binary_model = build_binary_model(binary_price_data, 20)
    binary_test_score = binary_model.score(binary_price_data['x_test'], binary_price_data['y_test'])
    print('\nBinary Test accuracy:', binary_test_score)

    pyplot.plot(
        regression_model.predict(regression_price_data['x_test']),
        label="prediction",
    )
    pyplot.plot(
        regression_price_data['y_test'],
        label="actual",
    )
    pyplot.legend()
    pyplot.show()



if __name__ == "__main__":
    args = get_args()
    if args.symbol:
        model_prediction(args.symbol)
        exit(1)
    raise ValueError("Missing symbol")
import argparse
import pandas as pd


from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, concatenate
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


def series_to_sl(values):
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
        x.append(input_data[start:end])
    for start, end in zip(range(0, len(label) - PREDICTION_STEP), range(PREDICTION_STEP, len(label))):
        y.append(label[start:end][0])

    x = np.array(x)
    y = np.array(y)

    return x, y, y_scaler


def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, header=0, index_col=0)
    values = data.to_numpy()
    return values.astype("float32")


def prepare_data(filepath, shuffle=True):
    data_values = load_data_from_csv(filepath)
    input_data, label, y_scaler = series_to_sl(data_values)
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, label, test_size=0.25, shuffle=shuffle)
    return x_train, x_test, y_train, y_test, y_scaler

def build_covid_model(x_train, y_train, batch_size, epochs,): 
    covid_model = Sequential()
    covid_data = x_train[:,:, 3:len(x_train[0][0])-1]
    covid_model.add(LSTM(LSTM_OUTPUT_SIZE, input_shape=(covid_data.shape[1:]), return_sequences=True))
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(LSTM(LSTM_OUTPUT_SIZE,return_sequences=False))
    covid_model.add(Dropout(0.2))
    covid_model.add(BatchNormalization())

    covid_model.add(Dense(DENSE1_OUTPUT_SIZE, activation='relu'))
    covid_model.add(Dense(DENSE2_OUTPUT_SIZE, activation='sigmoid'))

    covid_model.compile(loss='mse', optimizer='adam')

    covid_model.fit(covid_data, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
    return covid_model


def build_price_model(x_train, y_train, batch_size, epochs,): 
    price_model = Sequential()
    price_data = x_train[:,:, np.r_[0:3, 5]]

    price_model.add(LSTM(LSTM_OUTPUT_SIZE, input_shape=(price_data.shape[1:]), return_sequences=False))
    price_model.add(Dropout(0.2))
    price_model.add(BatchNormalization())

    # price_model.add(LSTM(LSTM_OUTPUT_SIZE,return_sequences=False))
    # price_model.add(Dropout(0.2))
    # price_model.add(BatchNormalization())

    price_model.add(Dense(DENSE1_OUTPUT_SIZE, activation='relu'))
    price_model.add(Dense(DENSE2_OUTPUT_SIZE, activation='sigmoid'))

    price_model.compile(loss='mse', optimizer='adam')

    price_model.fit(price_data, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
    return price_model


def build_overall_model(x_train, y_train, x_test, y_test, batch_size, epochs,):
    #covid_model 

    input_models = [build_covid_model(x_train, y_train, batch_size, epochs), build_price_model(x_train, y_train, batch_size, epochs)]
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

    hidden = Dense(DENSE2_OUTPUT_SIZE, activation='relu')(merge)
    output = Dense(DENSE3_OUTPUT_SIZE, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_inputs, outputs=output)

    model.compile(loss="mse", optimizer="adam")
    # overall_overall_model = Sequential()
    # overall_model.add(
    #     LSTM(LSTM_OUTPUT_SIZE, input_shape=(x_train.shape[1:]), return_sequences=True, stateful=False))
    # overall_model.add(Dropout(0.1))
    # overall_model.add(BatchNormalization())

    # overall_model.add(LSTM(LSTM_OUTPUT_SIZE, return_sequences=False,))
    # overall_model.add(Dropout(0.2))
    # overall_model.add(BatchNormalization())

    # overall_model.add(Dense(DENSE1_OUTPUT_SIZE, activation='relu'))
    # overall_model.add(Dense(DENSE2_OUTPUT_SIZE, activation='relu'))
    # overall_model.add(Dense(DENSE3_OUTPUT_SIZE, activation='softmax'))

    # overall_model.compile(loss='binary_crossentropy', optimizer='adam')

    # overall_model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, epochs=epochs)
    # score = overall_model.evaluate(x_test, y_test)
    # print("Validation accuracy percentage", score*100)

    return model

def fit_overall_model(model, x_train, y_train, epochs):
    covid_input = x_train[:,:, 3:len(x_train[0][0])-1]
    price_input = x_train[:,:, np.r_[0:3, 5]]
	# fit model
    model.fit([covid_input, price_input], y_train, epochs=epochs, verbose=0, shuffle=True)

# make a prediction with a stacked model
def predict_overall_model(model, x_train):
	# prepare input data
    covid_input = x_train[:,:, 3:len(x_train[0][0])-1]
    price_input = x_train[:,:, np.r_[0:3, 5]]
	# make prediction
    return model.predict([covid_input, price_input], verbose=0)


def model_prediction(symbol):
    prediction = []
    for i in range(0, 10):
        shuffle = True if i % 2 == 0 else False
        x_train, x_test, y_train, y_test, scaler = prepare_data(
            f"{symbol}_daily.csv", shuffle=shuffle)

        model = build_overall_model(x_train, y_train, x_test, y_test, batch_size=10, epochs=100,)
        fit_overall_model(model, x_test, y_test, epochs=100)
        
        x_train, x_test, y_train, y_test, scaler = prepare_data(
            f"{symbol}_daily.csv", shuffle=False)
        
        if i == 0 :
            prediction = predict_overall_model(model, x_test)
        else:
            prediction += predict_overall_model(model, x_test)
    # _, x_test_no_shuffle, _, y_test_no_shuffle, y_scaler = prepare_data(
    #     f"{symbol}_daily.csv", shuffle=False)

    
    # prediction = model.predict(x_test_no_shuffle, batch_size=10)
    prediction = np.array(scaler.inverse_transform(prediction/10))
    # mean = prediction.mean()
    # prediction -= mean
    # prediction *= 100
    # prediction += mean
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

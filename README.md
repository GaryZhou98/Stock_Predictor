# Stock Price Predictor Using LSTM Multi-head Model
#### Disclaimer: This is a project out of personal and academic interest to explore the use of deep learning models in real-world application. Not for commercial uses or any investment strategies.
#### Developers: Gary Zhou, Edward Saltzman

**- Project Overview:** Given the unprecedented Covid-19 pandemic that has swept across the U.S. since the beginning of 2020, the economic damage and uncertainties has greatly increased the volatility in the U.S. stock market. First came the huge crash in March, followed by four months of steady climb of major indices that saw a divergence in movements of the technology sector from other sectors that are more impacted by the pandemic. In the project, we hope to use traditional technical indicators, as well as Covid-19 data in the U.S., to train a LSTM model that could accurately, consistently predict the future movements and prices of any given stock using historical data. 

**- Model:** Since the prices of a stock is time-series data, we have decided to use a RNN model to forecast future prices. Our model consists of two parts: the first part has two individual models with LSTM and Dense layers, one to be trained using historical technical stock indicators and one trained using Covid data in U.S.; the second part then concatenates the outputs of the two individual models to further feed through Dense layers to produce the final prediction. The construction of the model is done by using Tensorflow2 and Keras.

**- Data:** The financial data is gathered, formatted, and saved as a CSV file for each individual stock using the Alpha Vantage API. The Covid-19 data is gathered, formatted, and saved as a CSV file using the CovidTracking API.

**- Hyper-parameters:**
- TIME_STEP: The number of days of data to use for prediction
- PREDICTION_STEP: The number of days to predict

- LSTM_OUTPUT_SIZE: The output size of the LSTM layer
- DENSE1_OUTPUT_SIZE: The output size of the first Dense layer
- DENSE2_OUTPUT_SIZE: The output size of the second Dense layer
- DENSE3_OUTPUT_SIZE: The output size of the third Dense layer

- PRICE_EPOCH: The number of epochs for the individual financial data model to trian
- COVID_EPOCH: The number of epochs for the individual Covid data model to trian
- OVERALL_EPOCH: The number of epochs for the overall model to trian

- TEST_PORTION: What percentage of data is used to test the accuracy of the model
- OVERALL_TRAIN_PORTION: What percentage of data is used to train the overall model. To increase accuracy and avoid overfitting, the two different type of training data for the model are both separated into two sets, one to be used to train the individual models and one for the overall model.

- TRAIN_SHUFFLE: Whether to shuffle the input data after each epoch during training
- SAMPLE_SHUFFLE: Whether to shuffle the data when spliting them into training and testing portions.

- NUM_TRAIN: The number of times to rebuild the models and produce new predictions. When NUM_TRAIN > 1, multiple copies of the model is produced and the prediction is accumulated. The final prediciton is the average of the accumulated predictions.

**- Current Accuracy:**
By plotting out the prediciton of our model and the actual labels using the data, it seems like our model could capture the trends in the movement of a particular stock's prices even for time periods during the pandemic. However, the absolute value of the predicted price deviates by a considerable amount.

**- Obstacles:**
1. Our current biggest obstacle is finding ways to further improve our models given the constraint of the limited Covid-19 data we have. Although it feels like eons since the pandemic and quarantine have begun, from a data perspective 6 months worth of data is insufficient when training a time-series model to forecast the future. Currently, to counter this issue, we simply concatenate all the historical prices available for a stock with the Covid data we have and fill all the days without Covid data with 0. However, this means that the model is mostly training on zeros which is fairly useless, a defect reflected the lack of decrease in loss during the training of the model. We need to find a way to be able to train our model given the limited data and improve accuracy.

2. On the other hand, the amount of historical financial data is quite large and thus the training time for our model has increased substantially when using our own local machine. Therefore, a migration to online cloud platforms such as GCP might be needed in the near future.




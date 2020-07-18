import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def series_to_sl(values, t_back, t_forward):
    df = pd.DataFrame(values)
    df.dropna(inplace=True)
    print('hello')
    concatted = pd.DataFrame()
    for shift in range(-t_back, t_forward):
        shifted = df.shift(-shift)
        if shift>=0:
            shifted = shifted.iloc[:,0:1]
        cols = []
        for i in range(shifted.shape[1]):
            cols.append(f'variable{i} t{shift}')
        shifted.columns = cols 
        concatted = pd.concat([concatted, shifted], axis=1)
    concatted.dropna(inplace=True)
    return concatted


def load_data_from_csv(filepath):
    data = pd.read_csv(filepath, header=0, index_col=0)
    values = data.values
    return values.astype('float32')


def prepare_data(filepath):
    data_values = load_data_from_csv(filepath) 
    df = series_to_sl(data_values, 3, 1)
    normalized_df = MinMaxScaler(feature_range=(0,1)).fit_transform(df)
   

prepare_data('FB_daily.csv') 

# model constants
TIME_STEP = 50
PREDICTION_STEP = 20

LSTM_OUTPUT_SIZE = 128
DENSE1_OUTPUT_SIZE = 64
DENSE2_OUTPUT_SIZE = 32
DENSE3_OUTPUT_SIZE = PREDICTION_STEP

COVID_EPOCH = 1
PRICE_EPOCH = 1
OVERALL_EPOCH = 1

TEST_PORTION = 0.1
OVERALL_TRAIN_PORTION = 0.4

TRAIN_SHUFFLE = True
SAMPLE_SHUFFLE = False

NUM_TRAIN = 1


# api constants
CREDENTIALS = "credentials.json"
COVID_COLUMNS = [
    "date",
    "positive",
    "negative",
    "death",
    "hospitalized",
    "deathIncrease",
    "hospitalizedIncrease",
    "negativeIncrease",
    "positiveIncrease",
]
COVID_CSV_PATH = "https://covidtracking.com/api/v1/us/daily.csv"
RECOMMENDATION_TRENDS_PATH = (
    "https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}"
)
RECOMMENDATION_COLUMNS = ["date", "buy", "hold", "sell"]
NEWS_SENTIMENT_PATH = "https://finnhub.io/api/v1/news-sentiment?symbol={}&token={}"
DAILY_PRICES_PATH = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}"

PRICE_INDICATORS = {
    "ATR": "https://www.alphavantage.co/query?function=ATR&symbol={}&interval=daily&time_period=14&apikey={}",
    "SMA": "https://www.alphavantage.co/query?function=SMA&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "EMA": "https://www.alphavantage.co/query?function=EMA&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "MOM": "https://www.alphavantage.co/query?function=MOM&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "MACD": "https://www.alphavantage.co/query?function=MACD&symbol={}&interval=daily&series_type=close&apikey={}",
    "STOCH": "https://www.alphavantage.co/query?function=STOCH&symbol={}&interval=daily&apikey={}",
    "RSI": "https://www.alphavantage.co/query?function=RSI&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "ADX": "https://www.alphavantage.co/query?function=ADX&symbol={}&interval=daily&time_period=10&apikey={}",
    "PLUS_DM": "https://www.alphavantage.co/query?function=PLUS_DM&symbol={}&interval=daily&time_period=10&apikey={}",
    "BBANDS": "https://www.alphavantage.co/query?function=BBANDS&symbol={}&interval=daily&time_period=5&series_type=close&nbdevup=3&nbdevdn=3&apikey={}",
    "OBV": "https://www.alphavantage.co/query?function=OBV&symbol=IBM&interval=daily&apikey={}",
    "CCI": "https://www.alphavantage.co/query?function=CCI&symbol={}&interval=daily&time_period=10&apikey={}",
    "AROON": "https://www.alphavantage.co/query?function=AROON&symbol={}&interval=daily&time_period=14&apikey={}",
}


UNEMPLOYMENT_SERIES_ID = "LNS14000000"  # seasonally adj. from BLS website
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


TIME_FORMAT = "%Y-%m-%d"
SAVED_CSV_PATH = "{}_daily.csv"
COVID_DATA_PATH = "covid.csv"

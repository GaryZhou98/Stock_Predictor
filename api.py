# from iexfinance.stocks import get_historical_data
from datetime import datetime, timedelta
import csv
import argparse
import csv
import datetime
import json
import pandas as pd
import numpy as np
import requests
import time

COVID_CSV_PATH = "https://covidtracking.com/api/v1/us/daily.csv"
COVID_COLUMNS = ['date','positive','negative','death','hospitalized','deathIncrease','hospitalizedIncrease','negativeIncrease', 'positiveIncrease']
RECOMMENDATION_TRENDS_PATH = (
    "https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}"
)
NEWS_SENTIMENT_PATH = "https://finnhub.io/api/v1/news-sentiment?symbol={}&token={}"
DAILY_PRICES_PATH = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}"

PRICE_INDICATORS = {
    "ATR": "https://www.alphavantage.co/query?function=ATR&symbol={}&interval=daily&time_period=14&apikey={}",
    "SMA": "https://www.alphavantage.co/query?function=SMA&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "EMA": "https://www.alphavantage.co/query?function=EMA&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "MACD": "https://www.alphavantage.co/query?function=MACD&symbol={}&interval=daily&series_type=close&apikey={}",
    "STOCH": "https://www.alphavantage.co/query?function=STOCH&symbol={}&interval=daily&apikey={}",
    "RSI": "https://www.alphavantage.co/query?function=RSI&symbol={}&interval=daily&time_period=10&series_type=close&apikey={}",
    "ADX": "https://www.alphavantage.co/query?function=ADX&symbol={}&interval=daily&time_period=10&apikey={}",
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
recommendation_cols = ["date", "buy", "hold", "sell"]
credentials = json.load(open("credentials.json", "r"))
end = datetime.datetime.now()

# pe = fb.get_earnings(period='year', token="pk_69c9cac10e344939be9ee5694af27d49")[0]['actualEPS']
eps = 7.30  # TTM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--start_date",
        help="start date TIME_FORMAT",
        type=lambda s: datetime.datetime.strptime(s, TIME_FORMAT),
        default=(datetime.date.today() - timedelta(1)).strftime(TIME_FORMAT),
    )
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    parser.add_argument("-f", "--file", help="pull symbols from file", default=False)
    parser.add_argument(
        "-n", "--new_row", help="add new data to existing csv", action="store_false"
    )
    return parser.parse_args()


def get_row_from_csv(csv_fname):
    with open(csv_fname, "r", encoding="latin-1") as csv_in:
        for row in csv.reader(csv_in):
            yield row


def get_new_row(symbol, file):
    prev_price_data = pd.read_csv(file).tail(4)
    last_date = prev_price_data.tail(1)["date"].values[0]
    if last_date >= (datetime.date.today() - timedelta(1)).strftime(TIME_FORMAT):
        print(
            f"Newest data for {symbol} has already been appended, aborting operation."
        )
        return
    prev_price_data = prev_price_data["closePrice"].to_numpy().astype("float64")
    price = requests.get(PRICE_PATH.format(symbol, credentials["av_api_key"])).json()
    price = float(
        price["Time Series (Daily)"][
            (datetime.date.today() - timedelta(1)).strftime(TIME_FORMAT)
        ]["4. close"]
    )
    pe = price / eps
    simple_avg = 0
    total = 0
    for num in prev_price_data:
        total += float(num)
    simple_avg = (total + price) / 5
    price_data = {
        "date": [datetime.date.today() - timedelta(1)],
        "closePrice": [price],
        "simpleAvg": [simple_avg],
        "pe": [pe],
    }
    price_data = pd.DataFrame.from_dict(price_data)
    covid_data = get_covid_data(
        (datetime.date.today() - timedelta(1)).strftime(TIME_FORMAT)
    )
    recommendation_trends = requests.get(
        RECOMMENDATION_TRENDS_PATH.format(symbol, credentials["finnhub_api_key"])
    ).json()[0]
    recommendation_trends = pd.DataFrame(recommendation_trends, index=[0]).drop(
        ["period", "symbol"], axis=1
    )
    atr = get_atr(
        symbol, (datetime.date.today() - timedelta(1)).strftime(TIME_FORMAT)
    ).reset_index()
    all_data = price_data.merge(
        covid_data, how="inner", left_index=True, right_index=True
    )
    all_data = all_data.merge(
        recommendation_trends, how="outer", left_index=True, right_index=True
    )
    all_data = all_data.merge(atr, how="outer", left_index=True, right_index=True)
    all_data = all_data.fillna(method="backfill")
    all_data = all_data.drop(["date_y", "index", "date"], axis=1).rename(
        columns={"date_x": "date"}
    )
    all_data.to_csv(SAVED_CSV_PATH.format(symbol), index=False, mode="a", header=False)
    print(f"added new {symbol} data to {SAVED_CSV_PATH.format(symbol)}.")


def get_data(symbol, start_date):
    print(f"pulling historical data for {symbol}...")
    daily_prices = get_daily_prices(symbol)
    price_indicators = get_price_indicators(symbol)
    historical = daily_prices.merge(price_indicators, how="inner", on="date")
    historical.insert(1, "close", historical.pop("close"))
    historical.to_csv(SAVED_CSV_PATH.format(symbol), index=False)
    print(f"wrote {symbol} data to {SAVED_CSV_PATH.format(symbol)}.")
    covid_data = get_covid_data()
    covid_data.to_csv(COVID_DATA_PATH, index=False)
    print(f"wrote covid data data to {COVID_DATA_PATH}.")


def get_covid_data():
    data = pd.read_csv(COVID_CSV_PATH)
    data["date"] = pd.to_datetime(data["date"], format='%Y%m%d')
    data = data.sort_values(by='date', ascending=True)
    data.reset_index(inplace=True)
    data = data[COVID_COLUMNS]
    data.fillna(0, inplace=True)
    return data


def get_daily_prices(symbol):
    daily_prices = requests.get(
        DAILY_PRICES_PATH.format(symbol, credentials["av_api_key"])
    ).json()
    daily_prices = pd.DataFrame.from_dict(
        daily_prices["Time Series (Daily)"], orient="index"
    )
    daily_prices.drop(columns=["5. volume"], inplace=True)
    daily_prices.reset_index(inplace=True)
    daily_prices["index"] = pd.to_datetime(daily_prices["index"], format=TIME_FORMAT)
    daily_prices.rename(
        columns={
            "index": "date",
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
        },
        inplace=True,
    )
    daily_prices = daily_prices.iloc[::-1]
    return daily_prices


def get_price_indicators(symbol):
    print("maximum API requests per minute reached. waiting one minute...")
    time.sleep(60)
    indicators = pd.DataFrame(columns=["date"])
    i = 0
    for indicator, path in PRICE_INDICATORS.items():
        if i % 5 == 0:
            print(indicator)
            print("maximum API requests per minute reached. waiting one minute...")
            time.sleep(60)
        indicators = indicators.merge(
            get_price_indicator(symbol, indicator, path), how="outer", on="date"
        )
        i += 1
    print("maximum API requests per minute reached. waiting one minute...")
    time.sleep(60)
    return indicators.dropna()


def get_price_indicator(symbol, indicator, path):
    path = path.format(symbol, credentials["av_api_key"])
    p_i = requests.get(path).json()
    p_i = pd.DataFrame.from_dict(
        p_i[f"Technical Analysis: {indicator}"], orient="index"
    )
    p_i.reset_index(inplace=True)
    p_i["index"] = pd.to_datetime(p_i["index"], format=TIME_FORMAT)
    p_i.rename(columns={"index": "date", indicator: indicator.lower()}, inplace=True)
    return p_i


def get_unemployment(start_date):
    headers = {"Content-type": "application/json"}
    data = json.dumps(
        {
            "seriesid": [UNEMPLOYMENT_SERIES_ID],
            "startyear": start_date.year,
            "endyear": datetime.date.today().year,
        }
    )
    res = json.loads(requests.post(BLS_API_URL, data=data, headers=headers).text)
    df = pd.DataFrame.from_dict(res["Results"]["series"][0]["data"])
    df["period"] = pd.to_datetime(df["periodName"] + " " + df["year"])
    df.drop(columns=["year", "periodName", "latest", "footnotes"], inplace=True)
    index = pd.date_range(df["period"].min(), datetime.date.today())
    df = df.set_index("period").reindex(index, method="backfill")
    df = df.reset_index()
    df["date"] = df["index"]
    df.rename(columns={"value": "unemploymentRate"}, inplace=True)
    return df[df["date"] >= start_date][["date", "unemploymentRate"]]


def get_recommendation_trends(symbol, start_date):
    all_recs = requests.get(
        RECOMMENDATION_TRENDS_PATH.format(symbol, credentials["finnhub_api_key"])
    ).json()
    all_recs = pd.DataFrame.from_records(all_recs)
    first_day_of_month = start_date.replace(day=1)
    total_expected_recs = (
        (datetime.date.today().year - first_day_of_month.year) * 12
        + datetime.date.today().month
        - first_day_of_month.month
    )
    assert (
        all_recs.size >= total_expected_recs
    ), f"{symbol} does not have enough recommendation trends for requested time period."
    first_day_of_month = first_day_of_month.strftime(TIME_FORMAT)
    all_recs = all_recs[all_recs["period"] >= first_day_of_month]
    index = pd.date_range(all_recs["period"].min(), datetime.date.today())
    all_recs["date"] = pd.to_datetime(all_recs["period"], format=TIME_FORMAT)
    all_recs = all_recs.set_index("date").reindex(index, method="backfill")
    all_recs = all_recs.reset_index()
    all_recs["date"] = all_recs["index"]
    return all_recs[all_recs["date"] >= start_date][recommendation_cols]


def get_news_sentimenet(symbol, start_date):
    print("get_news_sentiment is NOT YET IMPLEMENTED")


if __name__ == "__main__":
    args = get_args()
    if args.new_row:
        if not args.symbol or not args.file:
            raise ValueError(
                "Missing start_date, symbol and filepath for existing csv to add new row"
            )
        get_new_row(args.symbol, args.file)
        exit(1)
    elif args.symbol:
        get_data(args.symbol, args.start_date)
        if args.file:
            symbols = iter(get_row_from_csv(args.file))
            for row in symbols:
                for symbol in row:
                    get_data(symbol, args.start_date)

from iexfinance.stocks import get_historical_data
from datetime import datetime
import csv 
import argparse
import csv
import datetime
import json
import pandas as pd
import requests

COVID_CSV_PATH = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv"
RECOMMENDATION_TRENDS_PATH = (
    "https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}"
)
NEWS_SENTIMENT_PATH = "https://finnhub.io/api/v1/news-sentiment?symbol={}&token={}"
ATR_PATH = "https://www.alphavantage.co/query?function=ATR&symbol={}&interval=daily&time_period={}&apikey={}"
TIME_FORMAT = "%Y-%m-%d"
SAVED_CSV_PATH = "{}_daily.csv"
recommendation_cols = ["date", "buy", "hold", "sell", "strongBuy", "strongSell"]
credentials = json.load(open("credentials.json", "r"))
end = datetime.datetime.now()

# pe = fb.get_earnings(period='year', token="pk_69c9cac10e344939be9ee5694af27d49")[0]['actualEPS']
eps = 7.30 #TTM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--start_date",
        help="start date TIME_FORMAT",
        type=lambda s: datetime.datetime.strptime(s, TIME_FORMAT),
        default="2020-01-20",
    )
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    parser.add_argument("-f", "--file", help="pull symbols from file", default=False)
    return parser.parse_args()


def get_row_from_csv(csv_fname):
    with open(csv_fname, "r", encoding="latin-1") as csv_in:
        for row in csv.reader(csv_in):
            yield row

def get_data(symbol, start_date):
    print(f"pulling historical data for {symbol}...")
    covid_data = get_covid_data(start_date)
    recommendation_trends = get_recommendation_trends(symbol, start_date)
    atr = get_atr(symbol, start_date)
    price = get_price_and_pe(symbol, start_date)
    # news_sentiment = get_news_sentiment(symbol, start_date)
    all_data = price.merge(covid_data, how="inner", on="date")
    all_data = all_data.merge(recommendation_trends, how="outer", on="date")
    all_data = all_data.merge(atr, how="outer", on="date")
    all_data = all_data.fillna(method="backfill")
    all_data.to_csv(SAVED_CSV_PATH.format(symbol), index=False)
    print(f"wrote {symbol} data to {SAVED_CSV_PATH.format(symbol)}.")


def get_covid_data(start_date):
    data = pd.read_csv(COVID_CSV_PATH)
    data["date"] = pd.to_datetime(data["date"], format=TIME_FORMAT)
    index = pd.date_range(start_date, data["date"].max())
    data = data.set_index("date").reindex(index, fill_value=0)
    return data.reset_index().rename(columns={"index": "date"})

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


def get_atr(symbol, start_date):
    atr = requests.get(ATR_PATH.format(symbol, 14, credentials["av_api_key"])).json()
    atr = pd.DataFrame.from_dict(atr["Technical Analysis: ATR"], orient="index")
    atr = atr.reset_index().rename(columns={"index": "date", "ATR": "atr"})
    atr["date"] = pd.to_datetime(atr["date"], format=TIME_FORMAT)
    return atr[atr["date"] >= start_date]

def get_price_and_pe(symbol, start_date):
  data = []
  counter = 1
  hist = get_historical_data(symbol, start_date, end, close_only=True, token=credentials["iex_token"])

  for day in hist:
    price = hist[day]['close']
    temp = [day, price]
    if counter == 1:
      temp.append(price)
    elif counter == 2:
      temp.append((price + data[counter - 2][1])/2)
    elif counter == 3:
      temp.append((price + data[counter - 2][1] + data[counter - 3][1])/3)
    elif counter == 4:
      temp.append((price + data[counter - 2][1] + data[counter - 3][1] + data[counter - 4][1])/4)
    else:
      temp.append((price + data[counter - 2][1] + data[counter - 3][1] + data[counter - 4][1] + data[counter - 5][1])/5)
    
    counter += 1
    temp.append(price/eps)
    data.append(temp)
  
  res = {}
  for row in data:
    res[row[0]] = row[1:]

  res = pd.DataFrame.from_dict(res, orient="index")
  res = res.reset_index().rename(columns={"index": "date", 0 : "closePrice", 1 : "simpleAvg", 2 : "pe"})
  res["date"] = pd.to_datetime(res["date"], format=TIME_FORMAT)
  return res


def get_news_sentimenet(symbol, start_date):
    print("get_news_sentiment is NOT YET IMPLEMENTED")


if __name__ == "__main__":
    args = get_args()
    if args.symbol:
        get_data(args.symbol, args.start_date)
    if args.file:
        symbols = iter(get_row_from_csv(args.file))
        for row in symbols:
            for symbol in row:
                get_data(symbol, args.start_date)





# print(fb.get_quote())
# print(fb.get_price())
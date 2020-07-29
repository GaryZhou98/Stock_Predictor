# from iexfinance.stocks import get_historical_data
import argparse
import csv
import datetime
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import constants as c

num_av_requests = 0
credentials = json.load(open(c.CREDENTIALS, "r"))
end = datetime.now()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", help="symbol to pull data on", default=False)
    parser.add_argument("-f", "--file", help="pull symbols from file", default=False)
    parser.add_argument(
        "-c", "--covid_only", help="only pull covid data", action="store_true"
    )
    return parser.parse_args()


def get_row_from_csv(csv_fname):
    with open(csv_fname, "r", encoding="latin-1") as csv_in:
        for row in csv.reader(csv_in):
            yield row


def make_av_request(path):
    global num_av_requests
    if num_av_requests % 5 == 0:
        print("maximum API requests per minute reached. waiting one minute...")
        time.sleep(60)
    num_av_requests += 1
    return requests.get(path).json()


def get_price_data(symbol):
    print(f"pulling historical data for {symbol}...")
    daily_prices = get_daily_prices(symbol)
    price_indicators = get_price_indicators(symbol)
    historical = daily_prices.merge(price_indicators, how="inner", on="date")
    historical.insert(1, "close", historical.pop("close"))
    historical.to_csv(c.SAVED_CSV_PATH.format(symbol), index=False)
    print(f"wrote {symbol} data to {c.SAVED_CSV_PATH.format(symbol)}.")


def get_covid_data():
    data = pd.read_csv(c.COVID_CSV_PATH)
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    data = data.sort_values(by="date", ascending=True)
    data.reset_index(inplace=True)
    data = data[c.COVID_COLUMNS]
    data.fillna(0, inplace=True)
    data.to_csv(c.COVID_DATA_PATH, index=False)
    print(f"wrote covid data data to {c.COVID_DATA_PATH}.")


def get_daily_prices(symbol):
    daily_prices = make_av_request(
        c.DAILY_PRICES_PATH.format(symbol, credentials["av_api_key"])
    )
    daily_prices = pd.DataFrame.from_dict(
        daily_prices["Time Series (Daily)"], orient="index"
    )
    daily_prices.drop(columns=["5. volume"], inplace=True)
    daily_prices.reset_index(inplace=True)
    daily_prices["index"] = pd.to_datetime(daily_prices["index"], format=c.TIME_FORMAT)
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
    indicators = pd.DataFrame(columns=["date"])
    for indicator, path in c.PRICE_INDICATORS.items():
        indicators = indicators.merge(
            get_price_indicator(symbol, indicator, path), how="outer", on="date"
        )
    return indicators.dropna()


def get_price_indicator(symbol, indicator, path):
    path = path.format(symbol, credentials["av_api_key"])
    p_i = make_av_request(path)
    p_i = pd.DataFrame.from_dict(
        p_i[f"Technical Analysis: {indicator}"], orient="index"
    )
    p_i.reset_index(inplace=True)
    p_i["index"] = pd.to_datetime(p_i["index"], format=c.TIME_FORMAT)
    p_i.rename(columns={"index": "date", indicator: indicator.lower()}, inplace=True)
    return p_i


def get_unemployment(start_date):
    headers = {"Content-type": "application/json"}
    data = json.dumps(
        {
            "seriesid": [c.UNEMPLOYMENT_SERIES_ID],
            "startyear": start_date.year,
            "endyear": datetime.date.today().year,
        }
    )
    res = json.loads(requests.post(c.BLS_API_URL, data=data, headers=headers).text)
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
        c.RECOMMENDATION_TRENDS_PATH.format(symbol, credentials["finnhub_api_key"])
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
    first_day_of_month = first_day_of_month.strftime(c.TIME_FORMAT)
    all_recs = all_recs[all_recs["period"] >= first_day_of_month]
    index = pd.date_range(all_recs["period"].min(), datetime.date.today())
    all_recs["date"] = pd.to_datetime(all_recs["period"], format=c.TIME_FORMAT)
    all_recs = all_recs.set_index("date").reindex(index, method="backfill")
    all_recs = all_recs.reset_index()
    all_recs["date"] = all_recs["index"]
    return all_recs[all_recs["date"] >= start_date][c.RECOMMENDATION_COLUMNS]


if __name__ == "__main__":
    args = get_args()
    if args.covid_only:
        get_covid_data()
    elif args.symbol:
        get_covid_data()
        get_price_data(args.symbol)
    elif args.file:
        get_covid_data()
        symbols = iter(get_row_from_csv(args.file))
        for row in symbols:
            for symbol in row:
                get_price_data(symbol.strip())

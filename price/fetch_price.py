from binance import Client
import requests
from datetime import datetime, timedelta
import polars as pl


def binance_perp_names() -> list[str]:
    exchange_info = Client().futures_exchange_info()
    return [
        symbol["symbol"]
        for symbol in exchange_info["symbols"]
        if symbol["contractType"] == "PERPETUAL"
    ][:50]


def extract_timestamp_close_price(
    symbol: str,
    data: list[list[str | int]],
) -> list[dict[str, float | int]]:
    return [
        {"timestamp": int(item[0]), f"{symbol}#close": float(item[4])} for item in data
    ]


def fetch_perp_candles(
    symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    interval: str,
    limit: int = 10,
) -> list:
    res = requests.get(
        url="https://fapi.binance.com/fapi/v1/continuousKlines",
        params={
            "pair": symbol,
            "startTime": int(start_timestamp),
            "endTime": int(end_timestamp),
            "contractType": "PERPETUAL",
            "interval": interval,
            "limit": limit,
        },
        timeout=60,
    )
    res.raise_for_status()  # Raise an exception for HTTP errors
    return extract_timestamp_close_price(symbol, res.json())


def fetch_perp_hist_data(
    symbol: str, start_date: str, end_date: str, interval: str
) -> list:
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    start_time = int(start_datetime.timestamp() * 1000)
    end_time = int(end_datetime.timestamp() * 1000)
    all_klines = []

    while start_time < end_time:
        klines = fetch_perp_candles(symbol, start_time, end_time, interval, limit=1500)
        if not klines:
            print("No klines returned, breaking loop.")
            break
        all_klines.extend(klines)
        start_time = klines[-1]["timestamp"] + 1  # Move to the next timestamp
    return all_klines


def get_perp_hist_df(symbol: str, interval: str):
    start_date = "2020-01-01"
    end_date_df = datetime.now().date() + timedelta(days=1)
    end_date = end_date_df.strftime("%Y-%m-%d")
    data = fetch_perp_hist_data(symbol, start_date, end_date, interval)
    return pl.DataFrame(data)


def combine_coins_df(interval: str) -> pl.DataFrame:
    coins_name = binance_perp_names()
    combined_df = None
    for name in coins_name:
        df = get_perp_hist_df(name, interval)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.join(df, on="timestamp", how="full")
            combined_df = combined_df.drop("timestamp_right")
    return combined_df if combined_df is not None else pl.DataFrame()


if __name__ == "__main__":
    close_price_df = combine_coins_df("1h")
    close_price_df.write_csv("price.csv")

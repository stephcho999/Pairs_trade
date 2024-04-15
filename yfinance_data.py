import yfinance as yf
import pandas as pd

def get_price_df(ticker1, ticker2):
    """Retrieves OHLC price data for asset pair from Yahoo Finance"""

    asset1 = yf.Ticker(ticker1)
    asset2 = yf.Ticker(ticker2)

    asset1_df = asset1.history(period="max", interval="1d")
    asset2_df = asset2.history(period="max", interval="1d")

    asset1_df = asset1_df[["Open", "High", "Low", "Close"]]
    asset2_df = asset2_df[["Open", "High", "Low", "Close"]]

    df = pd.merge(
        asset1_df,
        asset2_df,
        on="Date",
        how="inner",
        suffixes=["_{}".format(ticker1), "_{}".format(ticker2)],
    )

    return df

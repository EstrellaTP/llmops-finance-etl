"""
Data Extraction Module for the LLMOps Finance ETL pipeline.
This script is responsible for ingesting quantitative market data 
and qualitative news data from external APIs.
"""

import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
NEWS_API = os.getenv("NEWS_API_KEY")


def get_data(ticker):
    """
    Extracts historical stock market data for a given company.

    Args:
        ticker (str): The stock symbol of the company (e.g., 'AAPL', 'MSFT').

    Returns:
        pandas.DataFrame: A dataframe containing the 'Date', 'Close' price, 
        and 'Volume' of the stock over the last month.
    """
    try: 
        bussiness = yf.Ticker(ticker)
        data = bussiness.history(period = "1d")
        filt_data = data[["Close", "Volume"]].reset_index()
        filt_data["Ticker"] = ticker
        return filt_data
    
    except Exception as e:
        print(f"Error extracting financial data for {ticker}: {e}")
        return pd.DataFrame(columns = ["Date", "Close", "Volume"])


def get_news(topic):
    """
    Fetches recent news articles related to a specific topic using the News API.

    Args:
        topic (str): The keyword or search query for the news (e.g., 'Apple stock').

    Returns:
        dict: A JSON response parsed into a Python dictionary containing 
        the retrieved news articles and metadata.
    """
    try: 
        date_7_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        url = (f"https://newsapi.org/v2/everything?"
               f"q={topic}&"
               f"from={date_7_days_ago}&"
               f"sortBy=publishedAt&"
               f"pageSize=50&"
               f"apiKey={NEWS_API}")
        
        output = requests.get(url)
        return output.json()
    
    except Exception as e:
        print(f"Error extracting news: {e}")
        return {"status" : "error", "totalResults": 0, "articles": []}


if __name__ == "__main__":
    print("--- Mock extraction ---")
    print("Extracting Data...")
    
    stock_data = get_data("AAPL")
    print("\n--- Stock Data ---")
    print(stock_data.head())
    

    news = get_news("Apple stock")
    print("\n--- JSON News ---")
    print(news)
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from datetime import  timedelta, datetime

import transform
import load

load_dotenv()

NEWS_API = os.getenv("NEWS_API_KEY")

def past_news(topic, date):

    try:
        target_date = date.strftime('%Y-%m-%d')
        date_7_days_ago = (date - timedelta(days=7)).strftime('%Y-%m-%d')

        url = (f"https://newsapi.org/v2/everything?"
               f"q={topic}&"
               f"from={date_7_days_ago}&"
               f"to={target_date}&"
               f"sortBy=publishedAt&"
               f"pageSize=50&"
               f"apiKey={NEWS_API}")
        
        output = requests.get(url)
        return output.json()
    
    except Exception as e:
        print("Error on retrieving past news at backfilling.")
        return {"status": "error", "articles": []}
    
def get_historical_prices(ticker, days_back, end_date_str):
    try: 
        bussiness = yf.Ticker(ticker)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        start_date = (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
        data = bussiness.history(start=start_date, end=end_date_str)
        filt_data = data[["Close", "Volume"]].reset_index()
        filt_data["Ticker"] = ticker
        return filt_data

    except Exception as e:
        print("Error finding data")
        return pd.DataFrame(columns = ["Date", "Close", "Volume"])
    

def execute_backfilling_pipeline(ticker, topic, days_back, end_date_str):
    try: 
        print("Initializing backfill pipeline.")
        hist_data = get_historical_prices(ticker, days_back, end_date_str)
        
        for index, row in hist_data.iterrows():
            current_date = row['Date']
            print(f"\n Processing datetime: {current_date.strftime('%Y-%m-%d')}...")
            
            df_daily = pd.DataFrame([row])
            
            raw_news = past_news(topic, current_date)
            
            clean_data = transform.clean_financial_data(df_daily)
            score = transform.analyze_sentiment(raw_news)
            merged_data = transform.merge_data(clean_data, score)

            loading = load.load_to_bigquery(merged_data)

            print("Waiting 5 sec to prevent Error 429...")
            time.sleep(5)

        print(f"\nBackfilling completed for {ticker}!")

    except Exception as e:
        print("Error initializing backfill pipeline.")


if __name__ == "__main__":
        
    execute_backfilling_pipeline("AAPL", "Apple stock", 25, "2026-07-12")


        
        

        

    




    
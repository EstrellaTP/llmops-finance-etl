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

STOCKS_TO_TRACK = {
    # Materias Primas
    "GC=F": "Gold market",              
    "SI=F": "Silver market",            
    "HG=F": "Copper market",            
    "CL=F": "Crude oil market",         
    "NG=F": "Natural Gas market",       
    "ZC=F": "Corn market",              
    "ZW=F": "Wheat market",             
    "CC=F": "Cocoa market",             
    "KC=F": "Coffee market",            
    
    # Índices Globales
    "^GSPC": "S&P 500 index",           
    "^DJI": "Dow Jones index",          
    "^IXIC": "Nasdaq index",            
    "^N225": "Nikkei 225 index",        
    "^FTSE": "FTSE 100 index",          
    "^GDAXI": "DAX index",              
    "^IBEX": "IBEX 35 index",           
    
    # Automoción
    "TSLA": "Tesla company",
    "TM": "Toyota Motors",
    "F": "Ford Motor company",
    "GM": "General Motors",
    "RACE": "Ferrari company",
    "HMC": "Honda Motors",
    "VOW3.DE": "Volkswagen company",
    "MBG.DE": "Mercedes-Benz company",
    "STLA": "Stellantis company",
    
    # Energía
    "XOM": "ExxonMobil",
    "CVX": "Chevron company",
    "SHEL": "Shell oil company",
    "TTE": "TotalEnergies",
    "BP": "BP oil company",
    "ENPH": "Enphase Energy",           
    "NEE": "NextEra Energy",            
    "IBE.MC": "Iberdrola company",      
    
    # Tecnología e IA
    "AAPL": "Apple company",
    "MSFT": "Microsoft company",
    "GOOGL": "Alphabet Google",
    "NVDA": "Nvidia company",
    "TSM": "Taiwan Semiconductor",      
    "ASML": "ASML Holding",             
    "AMD": "AMD company",
    "INTC": "Intel company",
    
    # Finanzas
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "V": "Visa company",
    "MA": "Mastercard company",
    
    # Consumo
    "AMZN": "Amazon company",
    "WMT": "Walmart company",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola company",
    "PEP": "PepsiCo",
    "MCD": "McDonald's company",
    "NKE": "Nike company",
    
    # Salud
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly",                 
    "PFE": "Pfizer company",
    "MRK": "Merck company",
    
    # Entretenimiento y Videojuegos
    "DIS": "Walt Disney company",
    "NFLX": "Netflix company",
    "META": "Meta Facebook",
    "SONY": "Sony Group",               
    "NTDOY": "Nintendo company",        
    "EA": "Electronic Arts",            
    "TTWO": "Take-Two Interactive"      
}



if __name__ == "__main__":
    print("Initializing massive backfill process...")
    
    for ticker, topic in STOCKS_TO_TRACK.items():
        print(f" Launching backfill for: {ticker} - {topic}")
        
        execute_backfilling_pipeline(ticker, topic, 1, "2026-07-09")
        
        print(f"Cooling down API limits before next company...")
        time.sleep(10)


        
        

        

    




    
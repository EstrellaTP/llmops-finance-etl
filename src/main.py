import time

# We imported the original modules (Phases 1-3)
import src.extract as extract
import src.transform as transform
import src.load as load

# We import the inference module (Phase 4)
import src.models.predict as predict

# GCP & Model Configurations 
# GCP & Model Configurations
PROJECT_ID = "llmops-finance-etl"
DATASET_ID = "finance_etl_db"
SOURCE_TABLE = "daily_market_data"
TARGET_TABLE = "model_predictions" 
BUCKET_NAME = "llmops-finance-models" 

# Neural Network hyperparameters (Must match architecture.py)
INPUT_SIZE = 5 # daily_return, volatility_7d, log_vol_change, sentiment_momentum, RSI_14d
HIDDEN_SIZE = 64
NUM_LAYERS = 1
def run_etl_and_predict(ticker, topic, sector):
    print(f"\n--- Starting Pipeline for {ticker} ({sector}) ---")
    
    # PHASE 1: Extraction
    raw_fin_data = extract.get_data(ticker)
    raw_news = extract.get_news(topic)

    # PHASE 2: Transformation & LLMOps
    clean_fin_data = transform.clean_financial_data(raw_fin_data)
    sentiment_score = transform.analyze_sentiment(raw_news)
    final_df = transform.merge_data(clean_fin_data, sentiment_score)

    # PHASE 3: Load to Data Warehouse
    load.load_to_bigquery(final_df)
    print("ETL phase successfully finished. Data loaded to BigQuery.")

    # PHASE 4: Deep Learning Inference
    try:
        predict.run_daily_inference(
            project_id=PROJECT_ID, 
            dataset_id=DATASET_ID, 
            bucket_name=BUCKET_NAME,
            source_table=SOURCE_TABLE, 
            target_table=TARGET_TABLE, 
            symbol=ticker, 
            sector_name=sector,
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=NUM_LAYERS
        )
    except Exception as e:
        print(f"Inference skipped or failed for {ticker}. Reason: {e}")
        print("Note: If the model hasn't been trained yet, this is expected.")

    print(f"--- Pipeline completed for {ticker} ---\n")


# Refactored tracking dictionary for Sector Segmentation
STOCKS_TO_TRACK = {
    "Raw Materials": {
        "GC=F": "Gold market",              
        "SI=F": "Silver market",            
        "HG=F": "Copper market",            
        "CL=F": "Crude oil market",         
        "NG=F": "Natural Gas market",       
        "ZC=F": "Corn market",              
        "ZW=F": "Wheat market",             
        "CC=F": "Cocoa market",             
        "KC=F": "Coffee market",            
    },
    "Global Indexes": {
        "^GSPC": "S&P 500 index",           
        "^DJI": "Dow Jones index",          
        "^IXIC": "Nasdaq index",            
        "^N225": "Nikkei 225 index",        
        "^FTSE": "FTSE 100 index",          
        "^GDAXI": "DAX index",              
        "^IBEX": "IBEX 35 index",           
    },
    "Automotive": {
        "TSLA": "Tesla company",
        "TM": "Toyota Motors",
        "F": "Ford Motor company",
        "GM": "General Motors",
        "RACE": "Ferrari company",
        "HMC": "Honda Motors",
        "VOW3.DE": "Volkswagen company",
        "MBG.DE": "Mercedes-Benz company",
        "STLA": "Stellantis company",
    },
    "Energy": {
        "XOM": "ExxonMobil",
        "CVX": "Chevron company",
        "SHEL": "Shell oil company",
        "TTE": "TotalEnergies",
        "BP": "BP oil company",
        "ENPH": "Enphase Energy",           
        "NEE": "NextEra Energy",            
        "IBE.MC": "Iberdrola company",      
    },
    "Technology and AI": {
        "AAPL": "Apple company",
        "MSFT": "Microsoft company",
        "GOOGL": "Alphabet Google",
        "NVDA": "Nvidia company",
        "TSM": "Taiwan Semiconductor",      
        "ASML": "ASML Holding",             
        "AMD": "AMD company",
        "INTC": "Intel company",
    },
    "Finance": {
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "GS": "Goldman Sachs",
        "V": "Visa company",
        "MA": "Mastercard company",
    },
    "Consume": {
        "AMZN": "Amazon company",
        "WMT": "Walmart company",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola company",
        "PEP": "PepsiCo",
        "MCD": "McDonald's company",
        "NKE": "Nike company",
    },
    "Health": {
        "JNJ": "Johnson & Johnson",
        "LLY": "Eli Lilly",                 
        "PFE": "Pfizer company",
        "MRK": "Merck company",
    },
    "Entertainment and Videogames": {
        "DIS": "Walt Disney company",
        "NFLX": "Netflix company",
        "META": "Meta Facebook",
        "SONY": "Sony Group",               
        "NTDOY": "Nintendo company",        
        "EA": "Electronic Arts",            
        "TTWO": "Take-Two Interactive"      
    }
}

if __name__ == "__main__":
    print("System On - Starting Batch Process and Inference")
    
    # Iterate over sectors and their respective tickers
    for sector, stocks in STOCKS_TO_TRACK.items():
        for ticker, topic in stocks.items():
            
            run_etl_and_predict(ticker, topic, sector)
            
            print(f"Waiting 10 seconds to respect API limits for {ticker}...")
            time.sleep(10)
            
    print("Batch process completely finished.")
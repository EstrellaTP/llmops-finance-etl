import extract
import transform
import load

def run_etl_pipeline(ticker, topic):

    print("Starting pipeline")
    
    raw_fin_data = extract.get_data(ticker)
    raw_news = extract.get_news(topic)

    clean_fin_data = transform.clean_financial_data(raw_fin_data)
    sentiment_score = transform.analyze_sentiment(raw_news)

    final_df = transform.merge_data(clean_fin_data, sentiment_score)

    load.load_to_bigquery(final_df)

    print("Pipeline has successfully finished!")

STOCKS_TO_TRACK = {
    # Raw Materials
    "GC=F": "Gold market",              
    "SI=F": "Silver market",            
    "HG=F": "Copper market",            
    "CL=F": "Crude oil market",         
    "NG=F": "Natural Gas market",       
    "ZC=F": "Corn market",              
    "ZW=F": "Wheat market",             
    "CC=F": "Cocoa market",             
    "KC=F": "Coffee market",            
    
    # Global Indexes
    "^GSPC": "S&P 500 index",           
    "^DJI": "Dow Jones index",          
    "^IXIC": "Nasdaq index",            
    "^N225": "Nikkei 225 index",        
    "^FTSE": "FTSE 100 index",          
    "^GDAXI": "DAX index",              
    "^IBEX": "IBEX 35 index",           
    
    # Automotive
    "TSLA": "Tesla company",
    "TM": "Toyota Motors",
    "F": "Ford Motor company",
    "GM": "General Motors",
    "RACE": "Ferrari company",
    "HMC": "Honda Motors",
    "VOW3.DE": "Volkswagen company",
    "MBG.DE": "Mercedes-Benz company",
    "STLA": "Stellantis company",
    
    # Energy
    "XOM": "ExxonMobil",
    "CVX": "Chevron company",
    "SHEL": "Shell oil company",
    "TTE": "TotalEnergies",
    "BP": "BP oil company",
    "ENPH": "Enphase Energy",           
    "NEE": "NextEra Energy",            
    "IBE.MC": "Iberdrola company",      
    
    # Tecnology and AI
    "AAPL": "Apple company",
    "MSFT": "Microsoft company",
    "GOOGL": "Alphabet Google",
    "NVDA": "Nvidia company",
    "TSM": "Taiwan Semiconductor",      
    "ASML": "ASML Holding",             
    "AMD": "AMD company",
    "INTC": "Intel company",
    
    # Finance
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "V": "Visa company",
    "MA": "Mastercard company",
    
    # Consume
    "AMZN": "Amazon company",
    "WMT": "Walmart company",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola company",
    "PEP": "PepsiCo",
    "MCD": "McDonald's company",
    "NKE": "Nike company",
    
    # Health
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly",                 
    "PFE": "Pfizer company",
    "MRK": "Merck company",
    
    # Entretainment and Videogames
    "DIS": "Walt Disney company",
    "NFLX": "Netflix company",
    "META": "Meta Facebook",
    "SONY": "Sony Group",               
    "NTDOY": "Nintendo company",        
    "EA": "Electronic Arts",            
    "TTWO": "Take-Two Interactive"      
}



if __name__ == "__main__":
    print("On - Starting Batch Process")
    
    for ticker, topic in STOCKS_TO_TRACK.items():
        print(f" Processing: {ticker} - {topic}")
        
        
        run_etl_pipeline(ticker, topic)
        
        print(f"Waiting 10 seconds to respect API limits for {ticker}...")
        time.sleep(10)
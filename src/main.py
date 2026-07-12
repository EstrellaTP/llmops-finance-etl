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


if __name__ == "__main__":
    ticker = "AAPL"
    topic = "Apple stock"
    
    print("On")
    
    run_etl_pipeline(ticker, topic)
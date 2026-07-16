
import pandas as pd
import os
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

def load_to_bigquery(df):
    """
    Loads enriched financial data into Google BigQuery using an idempotent 'Delete & Append' strategy.

    Args:
        df (pd.DataFrame): The final dataframe containing 'Date', 'Ticker', 'Close', 'Volume', and 'Sentiment'.

    Returns:
        bool: True if the load operation was successful, False otherwise.
    """
    if df.empty:
        print("El DataFrame está vacío. Se omite la carga para este activo.")
        return False

    try:
        client_bq = bigquery.Client()
        print("GCP conection established")
        print(f"Connected project: {client_bq.project}")

        my_project = client_bq.project
        my_dataset = "finance_etl_db"
        my_table = "daily_market_data"

        end_table = f"{my_project}.{my_dataset}.{my_table}"

        current_ticker = df["Ticker"].iloc[0]
        current_date = df['Date'].iloc[0].strftime('%Y-%m-%d')
        delete_query = f"""
            DELETE FROM `{end_table}`
            WHERE Ticker = '{current_ticker}' AND DATE(Date) = '{current_date}'
        """
        print(f"Cleaning previous data from {current_ticker} for current date")

        #NOTE: DML operations (such as DELETE) require active billing in GCP.
        # As this is a Proof of Concept (PoC) running on the Free Tier, 
        # the execution is commented out to prevent errors. 
        # The logic is kept to demonstrate the design of an idempotent pipeline.
        """
        duplicates_deletion = client_bq.query(delete_query)

        duplicates_deletion.result()
        """
        

        config_task = bigquery.LoadJobConfig(
            schema = [
                bigquery.SchemaField("Date", "TIMESTAMP"),
                bigquery.SchemaField("Ticker", "STRING"),
                bigquery.SchemaField("Close", "FLOAT"),
                bigquery.SchemaField("Volume", "INTEGER"),
                bigquery.SchemaField("Sentiment", "FLOAT"),
            ],
            
            write_disposition = "WRITE_APPEND",

        )
        
        print(f"Config is ready to point at: {end_table} .")

        task = client_bq.load_table_from_dataframe(
            df, end_table, job_config = config_task
        )
        
        task.result()

        print(f"You have inserted {task.output_rows} rows in BigQuery.")

        return True

    except Exception as e:
        print(f"Error with Google Cloud: {e} .")

        return False


if __name__ == "__main__":
    print("--- Mock Data Insertion ---")
    
    
    mock_data = pd.DataFrame({
        'Date': pd.to_datetime(['2026-07-12']), 
        'Ticker': ['AAPL'],
        'Close': [155.2], 
        'Volume': [1200500],
        'Sentiment': [0.8]
    })
    
    success = load_to_bigquery(mock_data)
    if success:
        print("Mock data processed and architecture evaluated successfully.")
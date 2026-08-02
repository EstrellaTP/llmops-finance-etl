
import torch.nn
import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from datetime import datetime

# Assuming the architecture is correctly imported from the module
from src.models.architecture import FinancialGRU
# from src.features.build_features import build_all_features 
# (You might need to adapt the import path depending on your execution context)

def fetch_latest_window(project_id: str, dataset_id: str, table_id: str, symbol: str, lookback_days: int = 14) -> pd.DataFrame:
    """
    Downloads the strictly necessary lookback window (default 14 days) 
    for a specific symbol from BigQuery to generate today's prediction.
    """
    client = bigquery.Client(project=project_id)
    
    # Extract the last N days ordered chronologically
    query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT {lookback_days}
    """
    
    df = client.query(query).to_dataframe()
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}.")
        
    # Reverse the dataframe so it goes from oldest to newest (chronological order)
    df = df.iloc[::-1].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def prepare_inference_tensor(df: pd.DataFrame, scaler_path: str):
    """
    Applies feature engineering and scales the data using the PRE-TRAINED scaler.
    Transforms the 2D Pandas DataFrame into a 3D PyTorch Tensor (1, 14, num_features).
    """
    # NOTE: You should run your feature engineering functions here.
    # For inference, you CANNOT drop nulls resulting from shift(-7) target 
    # because we don't have the future target yet. We only need the features.
    
    # 1. Load the fitted scaler from the training phase
    scaler = joblib.load(scaler_path)
    
    feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'target']]
    
    # 2. Scale the features
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    # 3. Convert to 3D Tensor: (batch_size=1, sequence_length=14, num_features)
    tensor_data = torch.tensor(df[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    
    return tensor_data

def load_pretrained_model(weights_path: str, input_size: int, hidden_size: int, num_layers: int) -> nn.Module:
    """
    Instantiates the GRU architecture and loads the sector-specific learned weights.
    """
    model = FinancialGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(weights_path))
    model.eval() # Set to evaluation mode (disables dropout, etc.)
    return model

def predict_direction(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """
    Performs the forward pass without tracking gradients to save memory and time.
    Returns the probability (0.0 to 1.0) of an uptrend in the next 7 days.
    """
    with torch.no_grad():
        probability = model(input_tensor)
        
    return probability.item()

def upload_prediction_to_bq(project_id: str, dataset_id: str, target_table: str, symbol: str, probability: float):
    """
    Stores the generated predictive signal into a derived BigQuery table.
    """
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{target_table}"
    
    # Determine binary signal based on a 0.5 threshold
    signal = 1 if probability > 0.5 else 0
    signal_label = "Uptrend" if signal == 1 else "Downtrend/Neutral"
    
    rows_to_insert = [
        {
            "execution_date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "symbol": symbol,
            "forecast_window": "7d",
            "uptrend_probability": probability,
            "signal": signal,
            "signal_label": signal_label
        }
    ]
    
    errors = client.insert_rows_json(table_ref, rows_to_insert)
    if not errors:
        print(f"Prediction for {symbol} successfully uploaded to {target_table}.")
    else:
        print(f"Errors occurred while uploading prediction: {errors}")

def run_daily_inference(project_id, dataset_id, source_table, target_table, symbol, weights_path, scaler_path, input_size, hidden_size, num_layers):
    """
    Main orchestrator for the daily continuous inference modality.
    """
    print(f"Starting daily inference for {symbol}...")
    
    # 1. Fetch data
    df_recent = fetch_latest_window(project_id, dataset_id, source_table, symbol, lookback_days=14)
    
    # 2. Prepare tensor
    input_tensor = prepare_inference_tensor(df_recent, scaler_path)
    
    # 3. Load Model
    model = load_pretrained_model(weights_path, input_size, hidden_size, num_layers)
    
    # 4. Predict
    prob = predict_direction(model, input_tensor)
    print(f"Predicted Uptrend Probability for {symbol}: {prob:.4f}")
    
    # 5. Upload signal
    upload_prediction_to_bq(project_id, dataset_id, target_table, symbol, prob)

if __name__ == "__main__":
    # Example execution trigger
    pass
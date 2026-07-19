
import pandas as pd
import numpy as np

# Data exctraction
def download_data_from_bq(df):
    pass


# Technical Variables
def daily_return(df):
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    return df


def volatility(df):
    df["volatility_7d"] = df.groupby("symbol")["daily_return"].rolling(window=7).std().reset_index(level=0, drop=True)
    return df


def RSI(df):
    delta = df.groupby('symbol')['close'].diff()
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gains = df.groupby('symbol')['gains'].rolling(window=14).mean().reset_index(level=0, drop=True)
    avg_losses = df.groupby('symbol')['losses'].rolling(window=14).mean().reset_index(level=0, drop=True)

    rs = avg_gains/avg_losses
    df['RSI_14d'] = 100 - (100 / (1 + rs))

    df = df.drop(columns=['gains', 'losses'])
    return df

def log_change_vol(df):
    prev_volume = df.groupby('symbol')['volume'].shift(1)
    
    df['log_vol_change'] = np.log(df['volume'] / prev_volume)

    return df


# Qualitative Variables
def sentiment_momentum(df):
    df["sentiment_momentum"] = df["sentiment"] - df.groupby("symbol")["sentiment"].rolling(window = 3).mean().reset_index(level=0,drop=True)
    return df

# Temporal varibles
def cyclic_calendar_features(df):

    day_of_week = pd.to_datetime(df['date']).dt.dayofweek
    
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 5)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 5)
    
    return df


def calculate_target(df):
    precio_futuro = df.groupby('symbol')['close'].shift(-7)
    
    df['target'] = np.where(precio_futuro > df['close'], 1, 0)
    
    return df


# Tensor transformation
def create_sequences(df, sequence_length=14):
    df = df.dropna()
    
    
    feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'target']]
    
    X = [] # 14 day matrices
    y = [] # Tarjet
    
    for symbol, group in df.groupby('symbol'):
        features = group[feature_cols].values
        target = group['target'].values
        
        
        for i in range(len(group) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
            
    return np.array(X), np.array(y)

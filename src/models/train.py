import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
import os
from google.cloud import storage

def upload_artifact_to_gcs(project_id: str, bucket_name: str, source_file_name: str, destination_blob_name: str):
    """
    Uploads a local file (model weights or scaler) to a Google Cloud Storage bucket.
    This ensures artifacts are kept out of version control.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    print(f"Artifact successfully uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")

def walk_forward_split(X, y, train_ratio=0.8):
    """
    Splits the dataset chronologically to prevent look-ahead bias.
    The training set must always precede the validation set.
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_val, y_val

def get_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Converts numpy arrays to PyTorch tensors and wraps them in DataLoaders
    for stable batch processing.
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # NEVER shuffle time series
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, scaler, project_id, bucket_name, epochs=50, learning_rate=0.001, sector_name="default"):
    """
    Trains the GRU model using Walk-Forward Validation and uploads the best 
    weights and the fitted scaler to Google Cloud Storage (MLOps artifact management).
    """
    # BCELoss expects probabilities between 0 and 1 (output of Sigmoid layer)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    # Define local temporary filenames for the artifacts
    weights_filename = f'weights_{sector_name}.pth'
    scaler_filename = f'scaler_{sector_name}.pkl'

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Save artifacts locally if current model achieves the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_filename)
            joblib.dump(scaler, scaler_filename)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
    print(f"Training cycle completed. Best Validation Loss: {best_val_loss:.4f}")
    
    # Upload the best local artifacts to Google Cloud Storage
    upload_artifact_to_gcs(project_id, bucket_name, weights_filename, f"models/{weights_filename}")
    upload_artifact_to_gcs(project_id, bucket_name, scaler_filename, f"models/{scaler_filename}")
    
    # Clean up local temporary files to keep the GitHub Actions runner clean
    if os.path.exists(weights_filename):
        os.remove(weights_filename)
    if os.path.exists(scaler_filename):
        os.remove(scaler_filename)
    
    return model
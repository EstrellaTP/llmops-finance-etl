import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

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

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, sector_name="default"):
    """
    Trains the GRU model using BCELoss and evaluates it chronologically 
    on the validation set to monitor overfitting.
    """
    # Using BCELoss because the FinancialGRU already applies a Sigmoid activation
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')

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
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Save the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'weights_{sector_name}.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    return model
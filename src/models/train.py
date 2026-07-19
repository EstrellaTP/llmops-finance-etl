import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train, epochs=50, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) 

    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        
        predictions = model(X_tensor)
        
        loss = criterion(predictions, y_tensor)
        
        loss.backward()
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Error (Loss): {loss.item():.4f}')
            
    return model
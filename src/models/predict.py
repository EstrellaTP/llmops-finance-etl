import torch

def predict_signal(model, recent_data):
    model.eval()
    
    X_tensor = torch.tensor(recent_data, dtype=torch.float32)
    
    with torch.no_grad():
        raw_output = model(X_tensor)
        
        probability = torch.sigmoid(raw_output).item()
        
    signal = 1 if probability > 0.5 else 0
    
    return signal, probability
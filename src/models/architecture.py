import torch
import torch.nn as nn

class FinancialGRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) neural network for binary time-series classification.
    Optimized for CPU execution and short lookback windows.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(FinancialGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Layer: Lighter and faster than LSTM, ideal for CPU training environments
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected layer to map the hidden state to a single output
        self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # x expected shape: (batch_size, sequence_length, num_features)
        
        # Forward pass through GRU
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.gru(x)
        
        # We only need the prediction from the last step of the sequence
        out = out[:, -1, :]
        
        # Pass the last state through the linear layer
        out = self.fc(out)
        
        # Output between 0 and 1 to get a probability
        out = self.sigmoid(out)
        
        return out
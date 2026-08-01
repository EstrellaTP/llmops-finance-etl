import numpy as np
from src.models.architecture import FinancialRNN
from src.models.train import train_model
from src.models.predict import predict_signal

def run_smoke_test():
    #Mock dimensions
    num_muestras = 100       
    sequence_length = 14     
    num_features = 5         

    print("Generating mock data...")
    X_dummy = np.random.rand(num_muestras, sequence_length, num_features)
    y_dummy = np.random.randint(0, 2, size=(num_muestras,))

    print("Installing neural network...")
    model = FinancialRNN(input_size=num_features, hidden_size=16)

    print("Training model...")
    model = train_model(model, X_dummy, y_dummy, epochs=2)

    print("Testing prediction...")
    X_recent = np.random.rand(1, sequence_length, num_features)
    signal, prob = predict_signal(model, X_recent)

    print("-" * 30)
    print("Success")
    print(f"Signal generated: {signal}")
    print(f"Probability of going up: {prob:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    run_smoke_test()
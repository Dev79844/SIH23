import pandas as pd
import numpy as np
import torch
from torch import nn

# Define the GRUModel class (same as during training)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def model(new_data_path:str):
    # Define the file paths for the trained model and new input data
    model_path = "cloud_burst_model.h5"
    # new_data_path = "test.csv"

    print(new_data_path)

    # Load the trained model
    model = torch.load('cloud_burst_model.h5')  # Make sure to match architecture
    # model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess new input data
    data = pd.read_csv(new_data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data['cld_cover'] = data['cld_cover'].values.reshape(-1, 1)
    sequence_length = 5  
    X = []
    print(data.head())

    # for i in range(len(data) - sequence_length):
    #     X.append(data.iloc[i:i+sequence_length]['cld_cover'].values)

    # X = torch.tensor(X, dtype=torch.float32)
    # X.reshape
    # print(X)
    # # Perform inference
    # with torch.no_grad():
    #     outputs = model(X.unsqueeze(1))
    #     _, predicted = torch.max(outputs, 1)

    # # Convert the predicted labels to class names if needed
    # class_names = ["low", "medium", "high"]
    # predicted_classes = [class_names[p] for p in predicted]

    # # Print or use the predicted classes as needed
    # print("Predicted Classes:")
    # print(predicted_classes)

    sequence_length = 5  # Adjust as needed
    X = []

    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length]['cld_cover'].values)

    X = torch.tensor(X, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        outputs = model(X.unsqueeze(2))
        _, predicted = torch.max(outputs, 1)

    # Convert the predicted labels to class names if needed
    class_names = ["low", "medium", "high"]
    predicted_classes = [class_names[p] for p in predicted]

    # Print or use the predicted classes as needed
    print("Predicted Classes:")
    print(predicted_classes)
    return predicted_classes

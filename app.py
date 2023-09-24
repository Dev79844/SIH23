from flask import Flask, request, jsonify
import inference
import pandas as pd
import numpy as np
import torch
from torch import nn

import firebase_admin


app = Flask(__name__)

firebase_admin.initialize_app(credential='services.json')

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

@app.route("/")
def predict():
    predictClass = inference.model('test3.csv')
    for val in predictClass:
        if val in ["high", "medium"]:
            return jsonify("Cloudburst")
    return jsonify("Not a cloudburst")


if __name__ == '__main__':
    app.run(port=3000, debug=True)


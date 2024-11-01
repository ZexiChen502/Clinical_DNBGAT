import json
from train import create_model
import torch
import os


def load(path):
    with open(os.path.join(path, 'model_params.json'), 'r') as file:
        model_params = json.load(file)

    loaded_model = create_model(**model_params)
    loaded_model.load_state_dict(torch.load('model.pth'))

    return load

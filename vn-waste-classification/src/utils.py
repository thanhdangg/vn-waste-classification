def load_model(model_path):
    import torch
    return torch.load(model_path)

def save_model(model, model_path):
    import torch
    torch.save(model, model_path)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(data_path):
    import pandas as pd
    return pd.read_csv(data_path)

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
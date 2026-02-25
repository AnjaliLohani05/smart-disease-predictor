import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import logging

# Load config
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'smart-disease-predictor', 'config', 'train_config.yaml'))
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['models']['tabular']

# Setup logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class TabularDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_model(config, model_name):
    logging.info(f"Training {model_name}")
    dataset_path = config[model_name]['dataset_path']
    save_path = config[model_name]['save_path']
    epochs = config[model_name]['epochs']
    batch_size = config[model_name]['batch_size']
    lr = config[model_name]['learning_rate']

    dataset = TabularDataset(dataset_path)
    input_dim = dataset.X.shape[1]
    num_classes = len(torch.unique(dataset.y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TabularMLP(input_dim, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"{model_name} Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
    torch.save(model.state_dict(), save_path)
    logging.info(f"{model_name} model saved to {save_path}")


def main():
    setup_logging()
    config = load_config()
    for model_name in config.keys():
        train_model(config, model_name)

if __name__ == '__main__':
    main()

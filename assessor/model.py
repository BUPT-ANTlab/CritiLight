import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv


class TrafficGNNDataset(Dataset):
    def __init__(self, data_path, graph_path):
        super().__init__()
        with open(graph_path, "rb") as handle:
            graph_info = pickle.load(handle)

        adj = graph_info["adj_matrix"]
        src, dst = np.nonzero(adj)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)

        with open(data_path, "rb") as handle:
            self.raw_samples = pickle.load(handle)

        print(f"Dataset loaded. Samples: {len(self.raw_samples)}")

    def len(self):
        return len(self.raw_samples)

    def get(self, idx):
        sample = self.raw_samples[idx]
        return Data(
            x=torch.tensor(sample["state_features"], dtype=torch.float32),
            edge_index=self.edge_index,
            y=torch.tensor(sample["prediction_labels"], dtype=torch.float32),
            action=torch.tensor(sample["action_index"], dtype=torch.float32),
            mask=torch.tensor(sample["masks"], dtype=torch.float32),
        )


class LightTrafficGAT(nn.Module):
    def __init__(self, state_dim=60, action_dim=12, output_dim=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        state_emb_dim = 64
        action_emb_dim = 64
        hidden_channels = 128

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gat1 = GATv2Conv(state_emb_dim + action_emb_dim, hidden_channels, heads=2, concat=True)
        self.gat2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, data):
        state_feat = self.state_encoder(data.x)
        action_feat = self.action_encoder(data.action)
        x = torch.cat([state_feat, action_feat], dim=-1)
        x = self.gat1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, data.edge_index)
        x = F.elu(x)
        return self.predictor(x)

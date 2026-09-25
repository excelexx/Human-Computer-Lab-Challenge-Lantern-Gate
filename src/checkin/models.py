"""Small learned heads: all three are included in the runtime parameter audit."""
import torch
from torch import nn

DIMENSIONS = {"vision": 1408, "text": 1024, "fusion": 2435}

class EmotionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 7))

    def forward(self, features):
        return self.net(features)

def load_head(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = EmotionHead(payload["input_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload

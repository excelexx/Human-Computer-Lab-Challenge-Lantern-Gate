"""Small learned heads: all three are included in the runtime parameter audit."""
from collections.abc import Mapping

import torch
from torch import nn

DIMENSIONS = {"vision": 1408, "text": 1024, "fusion": 2435}

class EmotionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 7))

    def forward(self, features):
        return self.net(features)


class LinearEmotionHead(nn.Module):
    """Affine seven-label classifier, with any training scaler already folded in."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, 7)

    def forward(self, features):
        return self.net(features)


def load_head(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Classifier checkpoint must contain a metadata mapping.")
    input_dim = payload.get("input_dim")
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("Classifier checkpoint input_dim must be a positive integer.")
    architecture = payload.get("architecture", "mlp")
    if architecture == "linear":
        model = LinearEmotionHead(input_dim)
    elif architecture == "mlp":
        hidden_dim = payload.get("hidden_dim", 128)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("MLP hidden_dim must be a positive integer.")
        model = EmotionHead(input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown classifier architecture: {architecture!r}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Classifier checkpoint must include a state_dict mapping.")
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Classifier weights must be finite floating-point tensors.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Classifier weights do not match the declared {architecture} architecture/input_dim: {error}") from error
    return model.to(device).eval(), payload

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TEMPORAL_ARCHITECTURE = "bilstm"
TEMPORAL_HIDDEN_SIZE = 73
TEMPORAL_NUM_LAYERS = 1
TEMPORAL_DROPOUT_RATE = 0.2174
TEMPORAL_LEARNING_RATE = 0.0004
TEMPORAL_BATCH_SIZE = 54
TEMPORAL_DEFAULT_EPOCHS = 73


@dataclass(frozen=True)
class TemporalRouterConfig:
    input_size: int
    label_count: int
    hidden_size: int = TEMPORAL_HIDDEN_SIZE
    num_layers: int = TEMPORAL_NUM_LAYERS
    bidirectional: bool = True
    dropout_rate: float = TEMPORAL_DROPOUT_RATE


def build_temporal_router_model(config: TemporalRouterConfig) -> Any:
    torch, nn = _torch_modules()

    class BiLSTMRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
            )
            direction_count = 2 if config.bidirectional else 1
            self.dropout = nn.Dropout(config.dropout_rate)
            self.head = nn.Linear(config.hidden_size * direction_count, config.label_count)

        def forward(self, inputs: Any) -> Any:
            _outputs, (hidden, _cell) = self.lstm(inputs)
            if config.bidirectional:
                pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                pooled = hidden[-1]
            return self.head(self.dropout(pooled))

    return BiLSTMRouter()


class TorchTemporalRouter:
    def __init__(
        self,
        *,
        checkpoint: dict[str, Any],
        labels: tuple[str, ...],
        device: str = "cpu",
    ) -> None:
        torch, _nn = _torch_modules()
        self.labels = labels
        self.device = torch.device(device)
        self.config = TemporalRouterConfig(
            input_size=int(checkpoint["input_size"]),
            label_count=len(labels),
            hidden_size=int(checkpoint.get("hidden_size", TEMPORAL_HIDDEN_SIZE)),
            num_layers=int(checkpoint.get("num_layers", TEMPORAL_NUM_LAYERS)),
            bidirectional=bool(checkpoint.get("bidirectional", True)),
            dropout_rate=float(checkpoint.get("dropout_rate", TEMPORAL_DROPOUT_RATE)),
        )
        self.model = build_temporal_router_model(self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        torch, _nn = _torch_modules()
        inputs = torch.tensor(values, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "torch is required to load temporal exercise router artifacts"
        ) from exc
    return torch, nn

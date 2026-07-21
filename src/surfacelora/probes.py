"""Independent post-hoc linear and nonlinear probes on frozen vectors."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import leakage_gap


class LinearProbe:
    def __init__(self, *, seed: int = 42, max_iter: int = 800, n_jobs: int | None = 16) -> None:
        import sklearn
        from sklearn.linear_model import LogisticRegression

        arguments = dict(
            solver="saga",
            max_iter=max_iter,
            tol=1e-3,
            random_state=seed,
        )
        parameters = inspect.signature(LogisticRegression).parameters
        if parameters["penalty"].default == "deprecated":
            arguments["l1_ratio"] = 0.0  # L2 under the current API.
        else:
            arguments["penalty"] = "l2"
        if "multi_class" in parameters:
            arguments["multi_class"] = "multinomial"
        version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
        if n_jobs is not None and version < (1, 8):
            arguments["n_jobs"] = int(n_jobs)
        self.model = LogisticRegression(**arguments)

    def fit(self, vectors: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        self.model.fit(np.asarray(vectors), np.asarray(labels))
        return self

    def accuracy(self, vectors: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.model.predict(vectors) == np.asarray(labels)))


class MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def probe_optimizer(probe: nn.Module, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.AdamW(list(probe.parameters()), lr=learning_rate, weight_decay=0.01)


def train_mlp_probe(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    validation_vectors: np.ndarray,
    validation_labels: np.ndarray,
    *,
    num_classes: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> MLPProbe:
    torch.manual_seed(seed)
    probe = MLPProbe(
        train_vectors.shape[1],
        num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = probe_optimizer(probe, learning_rate)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_vectors, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_x = torch.tensor(validation_vectors, dtype=torch.float32, device=device)
    validation_y = torch.tensor(validation_labels, dtype=torch.long, device=device)
    best_accuracy, best_state, stale = -1.0, None, 0
    for _ in range(max_epochs):
        probe.train()
        for vectors, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(probe(vectors.to(device)), labels.to(device)).backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            accuracy = float((probe(validation_x).argmax(-1) == validation_y).float().mean())
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            best_state = copy.deepcopy(probe.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP probe training produced no checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()
    return probe


@torch.inference_mode()
def mlp_accuracy(probe: MLPProbe, vectors: np.ndarray, labels: np.ndarray, device: str = "cpu") -> float:
    predictions = probe(torch.tensor(vectors, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()
    return float(np.mean(predictions == np.asarray(labels)))


def evaluate_probes(
    train_cache: Any,
    validation_cache: Any,
    test_cache: Any,
    *,
    config: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    settings = config.get("probes", {})
    chance = float(config.get("selection", {}).get("chance_accuracy", 0.20))
    seed = int(config.get("seed", 42))
    linear = LinearProbe(
        seed=seed,
        max_iter=int(settings.get("linear_max_iter", 800)),
        n_jobs=settings.get("linear_n_jobs", 16),
    ).fit(
        train_cache.vectors, train_cache.labels
    )
    linear_test = linear.accuracy(test_cache.vectors, test_cache.labels)
    mlp = train_mlp_probe(
        train_cache.vectors,
        train_cache.labels,
        validation_cache.vectors,
        validation_cache.labels,
        num_classes=len(train_cache.metadata.label_order),
        hidden_dim=int(settings.get("mlp_hidden_dim", 128)),
        dropout=float(settings.get("mlp_dropout", 0.3)),
        learning_rate=float(settings.get("mlp_learning_rate", 1e-3)),
        batch_size=int(settings.get("mlp_batch_size", 128)),
        max_epochs=int(settings.get("mlp_max_epochs", 50)),
        patience=int(settings.get("mlp_patience", 5)),
        seed=seed,
        device=device,
    )
    mlp_test = mlp_accuracy(mlp, test_cache.vectors, test_cache.labels, device)
    return {
        "representation": train_cache.metadata.representation,
        "layer": train_cache.metadata.layer,
        "checkpoint_identifier": train_cache.metadata.checkpoint_identifier,
        "linear_test_accuracy": linear_test,
        "linear_test_leakage_gap": leakage_gap(linear_test, chance),
        "mlp_test_accuracy": mlp_test,
        "mlp_test_leakage_gap": leakage_gap(mlp_test, chance),
        "n_train": int(len(train_cache.labels)),
        "n_validation": int(len(validation_cache.labels)),
        "n_test": int(len(test_cache.labels)),
    }


def evaluate_linear_validation(
    train_cache: Any,
    validation_cache: Any,
    *,
    config: dict[str, Any],
    rouge_l: float,
    step: int = 0,
) -> dict[str, Any]:
    """Produce one aggregate, selection-ready validation record without test access."""
    if not 0.0 <= float(rouge_l) <= 100.0:
        raise ValueError("rouge_l must use the 0-to-100 scale")
    settings = config.get("probes", {})
    chance = float(config.get("selection", {}).get("chance_accuracy", 0.20))
    linear = LinearProbe(
        seed=int(config.get("seed", 42)),
        max_iter=int(settings.get("linear_max_iter", 800)),
        n_jobs=settings.get("linear_n_jobs", 16),
    ).fit(train_cache.vectors, train_cache.labels)
    accuracy = linear.accuracy(validation_cache.vectors, validation_cache.labels)
    return {
        "checkpoint_identifier": train_cache.metadata.checkpoint_identifier,
        "split": "validation",
        "representation": train_cache.metadata.representation,
        "layer": train_cache.metadata.layer,
        "lr_probe_accuracy": accuracy,
        "lr_leakage_gap": leakage_gap(accuracy, chance),
        "rougeL": float(rouge_l),
        "step": int(step),
        "n_train": int(len(train_cache.labels)),
        "n_validation": int(len(validation_cache.labels)),
    }

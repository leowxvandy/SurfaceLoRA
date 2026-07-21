"""Training-time discriminator and trainable-parameter invariants."""

from __future__ import annotations

from torch import nn


class TrainingDiscriminator(nn.Module):
    """Default linear training-time sensitive-label discriminator."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, values):
        return self.classifier(values)


class MLPTrainingDiscriminator(nn.Module):
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

    def forward(self, values):
        return self.network(values)


def build_discriminator(
    architecture: str,
    input_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    if architecture == "linear":
        return TrainingDiscriminator(input_dim, num_classes)
    if architecture == "mlp":
        return MLPTrainingDiscriminator(
            input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout
        )
    raise ValueError("discriminator architecture must be 'linear' or 'mlp'")


def assert_trainable_isolation(language_model: nn.Module, discriminator: nn.Module) -> None:
    trainable_model = [name for name, parameter in language_model.named_parameters() if parameter.requires_grad]
    if not trainable_model:
        raise ValueError("No trainable LoRA parameters were found")
    unexpected = [name for name in trainable_model if "lora_" not in name.lower()]
    if unexpected:
        raise ValueError(f"Non-LoRA language-model parameters are trainable: {unexpected[:5]}")
    if not any(parameter.requires_grad for parameter in discriminator.parameters()):
        raise ValueError("The training-time discriminator has no trainable parameters")


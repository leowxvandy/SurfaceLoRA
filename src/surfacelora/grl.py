"""Gradient reversal with an identity forward pass."""

from __future__ import annotations

import torch


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(values: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    if coefficient < 0:
        raise ValueError("GRL coefficient must be non-negative")
    return _GradientReversal.apply(values, float(coefficient))


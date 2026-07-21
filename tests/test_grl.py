import torch

from surfacelora.grl import gradient_reverse


def test_gradient_reversal_preserves_forward_and_scales_negative_gradient():
    values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    output = gradient_reverse(values, 0.25)
    assert torch.equal(output, values)
    output.sum().backward()
    assert torch.allclose(values.grad, torch.full_like(values, -0.25))


def test_zero_coefficient_blocks_reverse_path():
    value = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(value, 0.0).sum().backward()
    assert torch.equal(value.grad, torch.zeros_like(value))

import math

import pytest
import torch

from src.objectives.sigreg import BHEP


def _manual_bhep(x: torch.Tensor, beta: float) -> torch.Tensor:
    n, d = x.shape
    beta2 = beta * beta

    squared_norm = x.square().sum(dim=1, keepdim=True)
    dist2 = squared_norm - 2.0 * (x @ x.T) + squared_norm.T
    dist2 = dist2.clamp_min(0.0)

    term1 = torch.exp(-dist2 / (2.0 * beta2)).mean()
    term2 = (
        2.0
        * math.exp(-0.5 * d * math.log1p(beta2))
        * torch.exp(-squared_norm.squeeze(1) / (2.0 * (1.0 + beta2))).mean()
    )
    term3 = math.exp(-0.5 * d * math.log1p(2.0 * beta2))
    return term1 - term2 + term3


def test_bhep_matches_closed_form_matrix_expression():
    x = torch.tensor(
        [
            [-1.0, 0.5],
            [0.0, -0.25],
            [1.25, 0.75],
        ],
        requires_grad=True,
    )
    beta = 0.7

    loss = BHEP(beta=beta)(x)
    expected = _manual_bhep(x, beta)

    assert torch.allclose(loss, expected, atol=1e-7)

    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_bhep_flattens_three_dimensional_inputs():
    x = torch.randn(4, 3, 2)

    three_dim = BHEP(beta=1.3)(x)
    flat = BHEP(beta=1.3)(x.reshape(12, 2))

    assert torch.allclose(three_dim, flat)


def test_bhep_scale_by_n_matches_unscaled_times_sample_count():
    x = torch.randn(5, 2, 3)

    unscaled = BHEP(beta=1.0, scale_by_n=False)(x)
    scaled = BHEP(beta=1.0, scale_by_n=True)(x)

    assert torch.allclose(scaled, unscaled * 10.0)


def test_bhep_rejects_nonpositive_beta():
    with pytest.raises(ValueError, match="beta must be positive"):
        BHEP(beta=0.0)

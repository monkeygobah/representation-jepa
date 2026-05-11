import math

import torch

from src.objectives.sigreg import EPPartial


def test_ep_partial_matches_three_term_formula_for_one_dimensional_input():
    x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], requires_grad=True)

    loss = EPPartial(num_slices=8)(x)

    x1 = x[:, 0]
    envelope = torch.exp(-0.5 * x1.square())
    c0 = envelope.mean()
    c1 = (envelope * x1).mean()
    c2 = (envelope * x1.square()).mean()
    expected = (
        (c0 - 1.0 / math.sqrt(2.0)).square()
        + c1.square()
        + 0.5 * (c2 - 1.0 / (2.0 * math.sqrt(2.0))).square()
    )

    assert torch.allclose(loss, expected, atol=1e-7)

    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_ep_partial_flattens_three_dimensional_inputs():
    x = torch.randn(4, 3, 1)

    torch.manual_seed(0)
    three_dim = EPPartial(num_slices=8)(x)
    torch.manual_seed(0)
    flat = EPPartial(num_slices=8)(x.reshape(12, 1))

    assert torch.allclose(three_dim, flat)


def test_ep_partial_scale_by_n_matches_unscaled_times_sample_count():
    x = torch.randn(5, 2, 1)

    torch.manual_seed(0)
    unscaled = EPPartial(num_slices=8, scale_by_n=False)(x)
    torch.manual_seed(0)
    scaled = EPPartial(num_slices=8, scale_by_n=True)(x)

    assert torch.allclose(scaled, unscaled * 10.0)

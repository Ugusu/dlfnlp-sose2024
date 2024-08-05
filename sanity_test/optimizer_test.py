import sys

import numpy as np
import torch

sys.path.append("../")
from optimizer import AdamW, SophiaG

seed = 0

"""
This script tests the correct functionality of the optimizer class by performing
an optimization task with AdamW.

Can only be called from same directory as the file.

The test is successful if the final weights of the model are close to the reference.
"""


def test_optimizer(opt_class: torch.optim.Optimizer) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    _ = torch.manual_seed(seed)

    # Create a simple model
    model = torch.nn.Linear(3, 2, bias=False)

    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        correct_bias=True,
    )

    # Optimize the model for a few steps
    for _ in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()

    return model.weight.detach()


ref = torch.tensor(np.load("optimizer_test.npy"))
actual = test_optimizer(AdamW)
print(ref)
print(actual)
assert torch.allclose(ref, actual, atol=1e-6, rtol=1e-4)
print("AdamW Optimizer test passed!")


# test sophia optimizer
def test_optimizer(opt_class: type[torch.optim.Optimizer]) -> torch.Tensor:
    seed = 42  # Define a seed for reproducibility
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Create a simple model
    model = torch.nn.Linear(3, 2, bias=False)

    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        rho=0.04,  # SophiaG specific parameter
        betas=(0.965, 0.99),  # SophiaG default betas
    )

    # Optimize the model for a few steps
    for _ in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()

        # Update Hessian before step (SophiaG specific)
        if isinstance(opt, SophiaG):
            opt.update_hessian()

        opt.step()

    return model.weight.detach()


# Run the test
print("Testing SophiaG optimizer...")
ref = torch.tensor(np.load("optimizer_test.npy"))  # Assuming this exists and contains the reference values
actual = test_optimizer(SophiaG)

print("Reference weights:")
print(ref)
print("Actual weights:")
print(actual)

# Compare the results
try:
    assert torch.allclose(ref, actual, atol=1e-6, rtol=1e-4)
    print("SophiaG optimizer test passed!")
except AssertionError:
    print("SophiaG optimizer test failed!")
    print("Absolute difference:")
    print(torch.abs(ref - actual))
    print("Relative difference:")
    print(torch.abs(ref - actual) / torch.abs(ref))

import torch


def baele(x):
    DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    return (
        (torch.tensor([1.5], device=DEVICE) - x[0] + x[0] * x[1]) ** 2
        + (torch.tensor([2.25], device=DEVICE) - x[0] + x[0] * x[1] ** 2) ** 2
        + (torch.tensor([2.625], device=DEVICE) - x[0] + x[0] * x[1] ** 3) ** 2
    )


def rosenbrock(x):
    DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    return (torch.tensor([1], device=DEVICE) - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

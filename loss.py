import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(- 10 * x))


def cdf(alpha, array):
    return 100 / len(array) * sigmoid(array - alpha).sum()


def total_variation(estimated_distribution, actual_distribution):
    loss = 0
    for step in torch.linspace(torch.min(actual_distribution), torch.max(actual_distribution), 500):
        loss += torch.square(cdf(step, actual_distribution) - cdf(step, estimated_distribution))
    return loss

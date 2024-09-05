import torch


def kth_moment(k):
    def moment(array):
        array_to_k = torch.pow(array, k)
        expected_to_k = torch.mean(array_to_k)
        return torch.pow(expected_to_k, 1/k)
    return moment


def moment_matching_loss(actual_distribution, estimated_distribution):
    loss = 0
    for k in range(1, 5):
        loss += torch.square(kth_moment(k)(actual_distribution) - kth_moment(k)(estimated_distribution))
    return loss

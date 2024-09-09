import matplotlib.pyplot as plt
import torch


def histogram(array, label, color):
    plt.hist(array, color=color, label=label, density=True, bins=500, histtype="stepfilled", alpha=.7)
    mean, std = torch.mean(array), torch.std(array)
    ymin, ymax = plt.ylim()
    plt.vlines([mean - std, mean, mean + std], colors=color, ymin=ymin, ymax=ymax)

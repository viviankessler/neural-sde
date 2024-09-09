import matplotlib.pyplot as plt
import torch


def histogram(array, label, color):
    array = array[~torch.isnan(array)]
    plt.hist(array, color=color, label=label, density=True, bins="auto", histtype="stepfilled", alpha=.7)
    mean, std = torch.mean(array), torch.std(array)
    plt.vlines([mean - std, mean, mean + std], colors=color, ymin=0, ymax=.1, linestyles="dotted")

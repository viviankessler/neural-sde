import matplotlib.pyplot as plt
import numpy as np


def histogram(array, label, color):
    plt.hist(array, color=color, label=label, density=True, bins="auto", histtype="stepfilled", alpha=.3)
    mean, std = np.mean(array), np.std(array)
    ymin, ymax = plt.ylim()
    plt.vlines([mean - std, mean, mean + std], colors=color, ymin=ymin, ymax=ymax)

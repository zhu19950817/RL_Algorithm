import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import brewer2mpl
import matplotlib as mpl
import time

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = cycler('color', bmap.mpl_colors)
params = {'axes.prop_cycle': colors}
mpl.rcParams.update(params)


def plot():
    start_time = time.time()
    score_list = np.loadtxt('CartPole.txt')
    plt.plot(score_list)
    x = np.array(range(len(score_list)))
    smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
    plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
    plt.ion()
    plt.pause(30)
    plt.close()


if __name__ == '__main__':
    while True:
        plot()
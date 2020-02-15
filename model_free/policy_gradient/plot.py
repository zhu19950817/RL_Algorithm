import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import brewer2mpl
import matplotlib as mpl

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = cycler('color', bmap.mpl_colors)
params = {'axes.prop_cycle': colors}
mpl.rcParams.update(params)


def plot():
    score_list = np.loadtxt('baseline.txt')
    plt.plot(score_list)
    x = np.array(range(len(score_list)))
    smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
    plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
    plt.show()


def Com_plot():
    score_list = np.loadtxt('baseline.txt')
    compare_list = np.loadtxt('Com-baseline.txt')
    plt.plot(score_list)
    plt.plot(compare_list)
    x = np.array(range(len(score_list)))
    smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
    y = np.array(range(len(compare_list)))
    smooth_com_func = np.poly1d(np.polyfit(y, compare_list, 3))
    plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
    plt.plot(y, smooth_com_func(y), label='Mean', linestyle='--')
    plt.show()


if __name__ == '__main__':
    Com_plot()
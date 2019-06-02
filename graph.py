import numpy as np
import time
import torch
import matplotlib
import matplotlib.pyplot as plt

def draw_graph(epoch_lst, accuracy_lst, model):

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_ylim([40, 100])
    ax.plot(epoch_lst, accuracy_lst, label='x: epoch, y: accuracy')
    plt.title(model)
    ax.legend(loc = 4)
    x = epoch_lst[np.argmax(accuracy_lst)]
    y = max(accuracy_lst)
    text= "epoch={:d}, acc={:.5f}".format(x, y)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(x, y), xytext=(0.94,0.96), **kw)
    fig.savefig('./result/' + model + '.png')

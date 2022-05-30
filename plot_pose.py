import matplotlib.pyplot as plt
import numpy as np


ARTIC_NAMES = {'foot 1': 0,
               'foot 2': 1,
               'knee 1': 2,
               'knee 2': 3,
               'belly': 4,
               'elbow 1': 5,
               'elbow 2': 6,
               'hand 1': 7,
               'hand 2': 8,
               'eye 1': 9,
               'eye 2': 10,
               'shoulder 1': 11,
               'shoulder 2': 12}

def plot_line(apsara, a1, a2, ax, c='r'):
    """
    Plot the leg/arm/... between articulations a1 and a2
    """
    i1 = ARTIC_NAMES[a1]
    i2 = ARTIC_NAMES[a2]
    ax.plot(apsara[(i1, i2), 0], apsara[(i1, i2), 1], c)
def plot_pose(apsara, ax=None, c='r'):
    if ax is None:
        fig, ax = plt.subplots()
    plt.gca().invert_yaxis()
    plot_line(apsara, 'foot 1', 'knee 1', ax, c=c)
    plot_line(apsara, 'knee 1', 'belly', ax, c=c)
    plot_line(apsara, 'foot 2', 'knee 2', ax, c=c)
    plot_line(apsara, 'knee 2', 'belly', ax, c=c)
    plot_line(apsara, 'shoulder 1', 'belly', ax, c=c)
    plot_line(apsara, 'shoulder 2', 'belly', ax, c=c)
    plot_line(apsara, 'shoulder 1', 'elbow 1', ax, c=c)
    plot_line(apsara, 'shoulder 2', 'elbow 2', ax, c=c)
    plot_line(apsara, 'elbow 1', 'hand 1', ax, c=c)
    plot_line(apsara, 'elbow 2', 'hand 2', ax, c=c)
    plot_line(apsara, 'shoulder 1', 'shoulder 2', ax, c=c)
    plot_line(apsara, 'eye 1', 'eye 2', ax, c=c)

def plot_all(apsaras, colors=None, labels=None):
    fig, ax = plt.subplots()
    ax.axis('off')
    line_length = int(np.sqrt(apsaras.shape[0]))
    if colors is None:
        for i, apsara in enumerate(apsaras):
            plot_pose(apsara + 5*np.array([[i%line_length, i//line_length]]), ax=ax)
    else:
        for i, apsara in enumerate(apsaras):
            plot_pose(apsara + 5*np.array([[i%line_length, i//line_length]]),
                      ax=ax,
                      c=colors(labels[i]))
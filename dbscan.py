from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as clstr
import matplotlib.pyplot as plt
import numpy as np
import itertools

def dbscan_choose_eps(data, chosen=0.):
    neighbors = NearestNeighbors(n_neighbors=2*data.shape[1])
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("Sorted by distance members of the database")
    ax.set_ylabel("Mutual average distance")
    ax.plot([0, data.shape[0]], [chosen, chosen], c='r')

def do_dbscan(data, eps, names):
    n, d = data.shape
    assert len(names) == d
    db = clstr.DBSCAN(eps=eps).fit(data)
    
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    # Pick the "core samples" and color them
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig, axs = plt.subplots(d, d, subplot_kw=dict(box_aspect=1),
                             sharex=True, constrained_layout=True)
    for i, j in itertools.product(range(d), range(d)):
        ax = axs[i, j]
        if i==j:
            ax.hist(data[:, i])
            ax.set(xlabel=names[i])
            continue
        elif i < j:
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k

                xy = data[class_member_mask & core_samples_mask]
                ax.plot(
                    xy[:, i],
                    xy[:, j],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

                xy = data[class_member_mask & ~core_samples_mask]
                ax.plot(
                    xy[:, i],
                    xy[:, j],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                )
                ax.set_ylim([0., np.pi])
            ax.set(xlabel=names[i], ylabel=names[j])
        else:
            ax.axis('off')

    fig.suptitle("Estimated number of clusters: %d" % n_clusters_)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()
    plt.show()
    
    return labels, [tuple(c) for c in colors + [[0, 0, 0, 1]]]
import numpy as np
import sklearn.cluster as clstr
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

N_CLUSTER = 16
def analyse_agls(agls_studied,
                 n_clusters=N_CLUSTER,
                 names=["knee 1", "knee 2"],
                 plot=True
                ):
    kma = clstr.KMeans(n_clusters=n_clusters,
                       max_iter=10000,
                       tol=1e-9
                      ).fit(agls_studied)
    centers = kma.cluster_centers_
    agls_frame = pd.DataFrame(agls_studied, columns=names)
    centers_frame = pd.DataFrame(centers, columns=names)
    results_frame = pd.concat([agls_frame,
                               centers_frame],
                              keys=['raw angles',
                                    'kmeans centers'],
                              names=['origin', 'number'])
    results_frame.reset_index(inplace=True)
    results_frame.drop(columns=['number'], inplace=True)
    if plot:
        sb.pairplot(results_frame, hue='origin')
    return results_frame, kma

def choose_km(agls_s,
              n_max,
              names=["knee 1", "knee 2"]
             ):
    assert n_max > 2, "choose more than 2 centers"
    sse = []
    itertor = list(range(2, n_max))
    for n in itertor:
        _, km = analyse_agls(agls_s,
                             n_clusters=n,
                             names=names,
                             plot=False
                            )
        sse.append(km.inertia_)
    plt.plot(itertor, sse)
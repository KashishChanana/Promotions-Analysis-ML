import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def elbow_method(df_clusters):
    """
    Plots two elbow method plots to check how many clusters inherently exist in the dataset
    :param df_clusters: dataframe with selected features
    :return: Displays charts indicating the number of clusters
    """
    distortions = []
    inertias = []

    # code cleaning and refactoring as a part of Advanced Software Engineering COMSW4156, taught by Prof. Gail Kaiser

    n_clusters = range(1, 10)

    for n in n_clusters:

        kmeanModel = KMeans(n_clusters=n).fit(df_clusters)
        kmeanModel.fit(df_clusters)

        distortions.append(sum(np.min(cdist(df_clusters, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / df_clusters.shape[0])
        inertias.append(kmeanModel.inertia_)

    plt.plot(n_clusters, distortions, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Squared Error (Cost)")
    plt.show()

    plt.plot(n_clusters, inertias, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Inertia")
    plt.show()
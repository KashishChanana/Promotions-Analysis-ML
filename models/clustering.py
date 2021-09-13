from sklearn.cluster import AgglomerativeClustering

def agglomerative_clustering(df_clusters, df):
    """
    Performs Hierarchical clustering on the dataframe
    :param df_clusters: dataframe with selected features
    :param df: master dataframe containing feature and target columns
    :return: Two clusters found after hierarchical clustering
    """
    # adjust n_clusters to the number obtained from elbow graphs

    # changes made as a part of Advanced Software Engineering COMSW4156, taught by Prof. Gail Kaiser
    n_clusters = 2

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df_clusters)
    df_clusters["label"] = cluster.labels_

    X_HAC_0 = df[df_clusters["label"] == 0]
    X_HAC_1 = df[df_clusters["label"] == 1]

    X_HAC_0.drop(columns=["success"], inplace=True)
    X_HAC_1.drop(columns=["success"], inplace=True)

    X_HAC_0.reset_index(drop=True)
    X_HAC_1.reset_index(drop=True)

    return X_HAC_0, X_HAC_1

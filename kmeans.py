# Base libraries
import numpy as np
import scipy as sp

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import streamlit as st

# Algorithm testing libraries
from sklearn.cluster import KMeans


# url = 'IRISDAT.TXT'
url = 'INCOME.csv'
df = pd.read_csv(url, sep='\t', comment='#') 

print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
df.head()

column_list = df.columns
for item in column_list:
    print(item)

# crim_lstat_array = np.array(df[['LISDLG', 'LISSZE']])
crim_lstat_array = np.array(df[['Aktyw', 'Przych']])

def recalculate_clusters(X, centroids, k):
    """ Recalculates the clusters """
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    # Setting the plot points using dataframe (X) and the vector norm (magnitude/length)
    for data in X:
        # Set up list of euclidian distance and iterate through
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - centroids[j]))
        # Append the cluster of data to the dictionary
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters    
 
def recalculate_centroids(centroids, clusters, k):
    """ Recalculates the centroid position based on the plot """ 
    for i in range(k):
        # Finds the average of the cluster at given index
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids

def plot_clusters(centroids, clusters, k):
    """ Plots the clusters with centroid and specified graph attributes """ 
    colors = ['red', 'blue' , 'green', 'orange', 'blue', 'gray', 'yellow', 'purple']
    fig = plt.figure(figsize = (6, 4))  
    area = (20) ** 2
    for i in range(k):
        for cluster in clusters[i]:
            plt.scatter(cluster[0], cluster[1], c=colors[i % k], alpha=0.6)          
        plt.scatter(centroids[i][0], centroids[i][1], c='black', s=200)
    st.write("fig")
    st.write(fig.figure)

def k_means_clustering(X, centroids={}, k=3, repeats=10):
    """ Calculates full k_means_clustering algorithm """
    for i in range(k):
        # Sets up the centroids based on the data
        centroids[i] = X[i]

    # Outputs the recalculated clusters and centroids 
    st.subheader('first and last iteration plot')
    print(f'First and last of {repeats} iterations')
    for i in range(repeats):        
        clusters = recalculate_clusters(X, centroids, k)  
        centroids = recalculate_centroids(centroids, clusters, k)

        # Plot the first and last iteration of k_means given the repeats specified
        # Default is 10, so this would output the 1st iteration and the 10th        
        if i == range(repeats)[-1] or i == range(repeats)[0]:
            plot_clusters(centroids, clusters, k)

    st.subheader("Clusters")
    st.write(clusters)
    for cluster in clusters:
      st.write(cluster)
    st.subheader("centroids")
    st.write(centroids)
    for cluster in centroids:
      st.write(cluster)


k_means_clustering(crim_lstat_array, k=3)

def sklearn_k_means(X, k=3, iterations=10):
    """ KMeans from the sklearn algorithm for comparison to algo from scratch """ 
    km = KMeans(
        n_clusters=k, init='random',
        n_init=iterations, max_iter=300, 
        random_state=0
    )

    y_km = km.fit_predict(X)

    plt.clf()

    # plot the 3 clusters
    fig1 = plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, 
        c='blue',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, 
        c='red',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, 
        c='green',
        label='cluster 3'
    )

    # plot the centroids
    for i in range(3):
        plt.scatter(
            km.cluster_centers_[-i, 0], km.cluster_centers_[-i, 1],
            s=100,
            c='black',
            label='centroids'
        )
    plt.legend(scatterpoints=1)
    plt.title('Clustered Data')
    plt.grid()
    plt.show()
    st.write("fig1")
    st.write(fig1.figure)

    plt.clf()

    # "Elbow Plot" to demonstrate what is essentially the usefulness of number for k
    # Calculate distortion (noise) for a range of number of cluster
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # Plot k vs distortion
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    fig2 = plt.ylabel('Distortion')
    plt.show()
    st.write("fig2")
    st.write(fig2.figure)

sklearn_k_means(crim_lstat_array)

# # plot 2-d with classes from df
# fig1 = plt.figure(figsize = (6, 4))  
# area = (20) ** 2
# plot_class = "Hrabstwo"
# df[plot_class] = pd.Categorical(df[plot_class])
# fig1 = df.plot.scatter(x="Aktyw", y="Przych", c=plot_class, cmap="viridis", s=50)  
# st.write(fig1.figure)



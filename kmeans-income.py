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

from sklearn.metrics.pairwise import manhattan_distances

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score
from scipy.spatial import distance
# distance.chebyshev([1, 0, 0], [0, 1, 0])
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# url = 'IRISDAT.TXT'
url = 'INCOME.csv'
df = pd.read_csv(url, sep=',', comment='#') 

all_res = []

print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
df.head()

column_list = df.columns
for item in column_list:
    print(item)

# crim_lstat_array = np.array(df[['LISDLG', 'LISSZE']])
crim_lstat_array = np.array(df[['Aktyw', 'Przych']])

def recalculate_clusters(X, centroids, k, dist):
    """ Recalculates the clusters """
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    # Setting the plot points using dataframe (X) and the vector norm (magnitude/length)
    for data in X:
        if(dist == "euc"):
            # Set up list of euclidian distance and iterate through
            euc_dist = []
            for j in range(k):
                euc_dist.append(np.linalg.norm(data - centroids[j]))
            # Append the cluster of data to the dictionary
            clusters[euc_dist.index(min(euc_dist))].append(data)
        if(dist == "l1"):
            # Set up list of l1 distance and iterate through
            l1_dist = []
            for j in range(k):
                l1_dist.append(manhattan_distances([data], [centroids[j]]))
            # Append the cluster of data to the dictionary
            clusters[l1_dist.index(min(l1_dist))].append(data)
        if(dist == "chebyshev"):
            # Set up list of chebyshev distance and iterate through
            chebyshev_dist = []
            for j in range(k):
                # chebyshev_dist.append(distance.chebyshev([data], [centroids[j]]))
                chebyshev_dist.append(np.max(np.abs(data - centroids[j])))
            # Append the cluster of data to the dictionary
            clusters[chebyshev_dist.index(min(chebyshev_dist))].append(data)
        if(dist == "mahal"):
            # Set up list of mahal distance and iterate through
            mahal_dist = []
            # iv = np.cov(X)
            iv = covMatrix
            for j in range(k):
                mahal_dist.append(distance.mahalanobis([data], [centroids[j]], iv))
            # Append the cluster of data to the dictionary
            clusters[mahal_dist.index(min(mahal_dist))].append(data)
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
        st.write(f'cluster:{i % k}, color:{colors[i % k]}')
        for cluster in clusters[i]:
            plt.scatter(cluster[0], cluster[1], c=colors[i % k], alpha=0.6)          
        plt.scatter(centroids[i][0], centroids[i][1], c='black', s=200)
    st.write("fig")
    st.write(fig.figure)

def k_means_clustering(X, centroids={}, k=3, repeats=10, dist="euc"):
    """ Calculates full k_means_clustering algorithm """
    for i in range(k):
        # Sets up the centroids based on the data
        centroids[i] = X[i]
        st.write(f"centroids: {centroids[i]}")

    # Outputs the recalculated clusters and centroids 
    st.subheader('first and last iteration plot')
    print(f'First and last of {repeats} iterations')
    for i in range(repeats):        
        clusters = recalculate_clusters(X, centroids, k, dist)  
        centroids = recalculate_centroids(centroids, clusters, k)

        # Plot the first and last iteration of k_means given the repeats specified
        # Default is 10, so this would output the 1st iteration and the 10th        
        if i == range(repeats)[-1] or i == range(repeats)[0]:
            plot_clusters(centroids, clusters, k)

    df1 = pd.DataFrame([ [key, len(value)] for key, value in clusters.items()], columns = ['Cluster', 'Number'])
    st.subheader(f"Clusters")
    st.write(df1)
    st.subheader("Classes")
    st.write(df['Hrabstwo'].value_counts()) 

    st.header("Results")
    
    # st.write(df)
    new_df = pd.DataFrame()

    for cluster in clusters:
        for index, value in enumerate(clusters[cluster]):
            df_with_cluster = pd.DataFrame.from_dict(clusters[cluster])
        df_with_cluster['cluster'] = cluster
        
        new_df = pd.concat([new_df, df_with_cluster], axis=0)
        new_df.reset_index(drop=True)
    
    df_to_join_1 = df.sort_values(by=['Aktyw', 'Przych']).reset_index(drop=True)
    df_to_join_2 = new_df.sort_values(by=[0, 1]).drop_duplicates().reset_index(drop=True)
    st.write(df_to_join_1)
    st.write(df_to_join_2)
    df_info = pd.merge(df_to_join_1, df_to_join_2,  how='left', left_on=['Aktyw', 'Przych'], right_on = [0, 1])

    # st.header('dane1')
    # st.write(len(df[['Aktyw','Przych']]))
    # st.write(len(new_df[[0,1]]))
    # st.write(len(df_col_merged))
    # st.write(df_col_merged)
    
    df_info["cluster2"] = df_info["cluster"].map({0: 'HIGHLAND', 1:'DODGE', 2:'ROGERS'}) 
    df_info["hrabstwo_to_num"] = df_info["Hrabstwo"].map({'HIGHLAND': 0, 'DODGE': 1, 'ROGERS':2}) 
    st.write(df_info["cluster2"], df_info["Hrabstwo"])

    # df_confusion = pd.crosstab(df_info["cluster2"], df_info["Hrabstwo"])
    df_confusion = pd.crosstab(df_info["cluster2"], df_info["Hrabstwo"], rownames=['Actual'], colnames=['Predicted'], margins=True)

    # st.write("df_confusion")
    st.write("Confusion matrix")
    st.dataframe(df_confusion)

    accuracy_sc = accuracy_score(df_info["cluster2"], df_info["Hrabstwo"])
    st.write(f'accuracy_score: {accuracy_sc}')

    jaccard_sc = jaccard_score(df_info["cluster2"], df_info["Hrabstwo"], average=None)
    st.write(f'jaccard_score: {jaccard_sc}')

    silhouette_sc = silhouette_score(df_info[["Aktyw","Przych","hrabstwo_to_num"]], df_info["cluster"])
    st.write(f'silhouette_sc: {silhouette_sc}')
    
    cosine_sc = cosine_similarity(np.asmatrix(df_info["cluster"]), np.asmatrix(df_info["hrabstwo_to_num"]))[0][0]
    st.write(f'cosine_similarity: {cosine_sc}')

    
    res = pd.DataFrame([{'metrics':dist, 'accuracy_score':accuracy_sc, 'jaccard_score':jaccard_sc, 'silhouette_score':silhouette_sc, 'cosine_similarity':cosine_sc}])
    st.write(res)
    all_res.append(res)


# results_df = pd.DataFrame(columns=['distance','accuracy_score', 'jaccard_score', 'silhouette_sc', 'cosine_similarity'])
# results_df= pd.DataFrame([{'distance':1, 'accuracy_score':1, 'jaccard_score':1, 'silhouette_sc':1, 'cosine_similarity':1}])

"""For 4 metrics"""
st.header("euclidean")
k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="euc")
plt.clf()

# st.header("l1")
# k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="l1")
# plt.clf()

st.header("chebyshev")
k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="chebyshev")
plt.clf()

# st.header("mahalanobis")
# data = np.array([df['Aktyw'],df['Przych']])
# covMatrix = np.cov(data,bias=True)
# st.write(covMatrix)
# k_means_clustering(crim_lstat_array, k=3, repeats=10, dist="mahal")
# plt.clf()


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

def elbow_plot(X, k=3, iterations=10):
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



# plot 2-d with classes from df
fig1 = plt.figure(figsize = (6, 4))  
area = (20) ** 2
plot_class = "Hrabstwo"
df[plot_class] = pd.Categorical(df[plot_class])
fig1 = df.plot.scatter(x="Aktyw", y="Przych", c=plot_class, cmap="viridis", s=50)  
st.write(fig1.figure)


# sklearn_k_means(crim_lstat_array)
elbow_plot(crim_lstat_array)


df_res = pd.concat(all_res)
st.write(df_res)
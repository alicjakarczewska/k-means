from KMeans_Class import KMeans
from sklearn import datasets
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt

st.header("Analiza skupien")

# We load the iris dataset to use it in this example. We take the save the data and the target in two different variables
iris = datasets.load_iris()
X = iris.data
y = iris.target

st.write(iris.head())

model = KMeans(k=5, maxiter =300, distance='cosine', record_heterogeneity=[], verbose=True, seed=123)

centroids, cluster_assignment = model.fit(X)

st.write(centroids)

st.write(cluster_assignment)

fig1 = model.plot_heterogeneity()

st.write(fig1.figure)


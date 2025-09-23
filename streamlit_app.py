import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
# Shared Data

df = pd.read_csv('https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/customer_data.csv')
st.title('Customer Segmentation Dashboard')
st.write('This dashboard allows you to perform customer segmentation using various clustering algorithms.')
st.write('Dataset Preview:')
st.dataframe(df.head())

scaler=StandardScaler()
pca=PCA()
pipeline= make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(df)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

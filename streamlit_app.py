import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline

# Shared Data

df = pd.read_csv('https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/customer_data.csv')
st.title('Customer Segmentation Dashboard')
st.write('This dashboard allows you to perform customer segmentation using various clustering algorithms.')
st.write('Dataset Preview:')
st.dataframe(df.head())

sc=StandardScaler()
pca=PCA()
pipeline=Pipeline([('scaler', sc), ('pca', pca)])

pipeline.fit(df)

features = range(pca.n_components_)_
plt.figure(figsize=(10, 6))
plt.plot(features, pca.explained_variance_ratio_, marker='o')
plt.title('Explained Variance by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram

# ==========================
# Load Data
# ==========================
df = pd.read_csv('https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/customer_data.csv')

st.title('Customer Segmentation Dashboard')
st.write('This dashboard allows you to perform customer segmentation using various clustering algorithms.')
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# Data Cleaning
# ==========================
df.dropna(inplace=True)
if 'CustomerID' in df.columns:
    df.drop(['CustomerID'], axis=1, inplace=True)
df = pd.get_dummies(df)

# ==========================
# Scale + PCA
# ==========================
scaler = StandardScaler()
pca = PCA(n_components=2)  # Reduce to 2D for visualization
pipeline = make_pipeline(scaler, pca)
X_pca = pipeline.fit_transform(df)

# Calculate the linkage: mergings
mergings = linkage(df,method='single')

# Plot the dendrogram
dendrogram(mergings,leaf_rotation=90,leaf_font_size=6)
plt.show()


# ==========================
# Sidebar - Select Clustering Method
# ==========================
st.sidebar.title("Clustering Options")
algo = st.sidebar.selectbox("Choose an algorithm:", ["KMeans", "DBSCAN", "Agglomerative"])

if algo == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif algo == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:  # Agglomerative
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=n_clusters)

# ==========================
# Fit Model & Predict
# ==========================
labels = model.fit_predict(X_pca)

# ==========================
# Evaluate Clustering
# ==========================
if len(set(labels)) > 1 and -1 not in labels:  # silhouette needs >1 cluster and no noise label -1
    score = silhouette_score(X_pca, labels)
    st.write(f"**Silhouette Score:** {score:.3f}")
else:
    st.write("Silhouette Score: Not applicable (only one cluster or DBSCAN noise detected).")

# ==========================
# PCA Variance Plot
# ==========================
st.subheader("PCA Explained Variance")
features = range(len(pca.explained_variance_))
fig, ax = plt.subplots()
ax.bar(features, pca.explained_variance_)
ax.set_xlabel('PCA Feature')
ax.set_ylabel('Variance')
ax.set_xticks(features)
st.pyplot(fig)

# ==========================
# Cluster Visualization
# ==========================
st.subheader("Cluster Visualization (PCA 2D)")
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

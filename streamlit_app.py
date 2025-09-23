import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline

# ==========================
# Load Data
# ==========================
df_raw = pd.read_csv('https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/customer_data.csv')

st.title('Customer Segmentation with KMeans')
st.write('This dashboard performs customer segmentation using **KMeans clustering**.')
st.subheader("Dataset Preview")
df=df_raw.drop(['purchase_history'],axis=1)
st.dataframe(df.head())


# ==========================
# Data Cleaning
# ==========================
df.dropna(inplace=True)
if 'CustomerID' in df.columns:
    df.drop(['CustomerID'], axis=1, inplace=True)


# One-hot encode for clustering
df_encoded = pd.get_dummies(df)
# ==========================
# Scale + PCA
# ==========================
scaler = StandardScaler()
pca = PCA(n_components=2)  # Reduce to 2D for visualization
pipeline = make_pipeline(scaler, pca)
X_pca = pipeline.fit_transform(df_encoded)


# ========================== 
# # Sidebar - KMeans Parameters 
# # ========================== 
st.sidebar.title("KMeans Options")
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3) 
model = KMeans(n_clusters=n_clusters, random_state=42)
# ==========================
# Fit Model & Predict
# ==========================
labels = model.fit_predict(X_pca)

# ==========================
# Crosstab (Clusters vs Purchase History)
# ==========================
if 'purchase_history' in df_raw.columns:
    crosstab = pd.crosstab(labels, df_raw['purchase_history'], 
                           rownames=['Cluster'], colnames=['Purchase History'])
    st.subheader("Cluster vs Purchase History")
    st.dataframe(crosstab)
else:
    st.error("purchase_history column not found in dataset.")


# ==========================
# Evaluate Clustering
# ==========================
if len(set(labels)) > 1:
    score = silhouette_score(X_pca, labels)
    st.write(f"**Silhouette Score:** {score:.3f}")

# ==========================
# Cluster Counts
# ==========================
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.subheader("Cluster Counts")
st.write(cluster_counts)

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


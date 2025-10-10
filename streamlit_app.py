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
# Page Configuration
# ==========================
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# ==========================
# Load Data
# ==========================
df_raw = pd.read_csv('https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/customer_data.csv')

st.title("🧩 Customer Segmentation with KMeans")
st.markdown(
    "This interactive dashboard segments customers into groups using **KMeans clustering**. "
    "Understanding these segments helps businesses improve marketing strategies and customer engagement."
)

# ==========================
# Dataset Preview
# ==========================
st.header("📋 Dataset Overview")
df = df_raw.drop(['purchase_history'], axis=1)
st.dataframe(df.head(), use_container_width=True)

# ==========================
# Data Cleaning
# ==========================
df.dropna(inplace=True)
if 'CustomerID' in df.columns:
    df.drop(['CustomerID'], axis=1, inplace=True)

# One-hot encode categorical data
df_encoded = pd.get_dummies(df)

# ==========================
# Data Scaling & PCA
# ==========================
scaler = StandardScaler()
pca = PCA(n_components=2)
pipeline = make_pipeline(scaler, pca)
X_pca = pipeline.fit_transform(df_encoded)

# ==========================
# Sidebar Controls
# ==========================
st.sidebar.title("⚙️ Model Options")
st.sidebar.markdown("💡 *Tip: KMeans performs best at **k = 4** based on silhouette analysis.*")
n_clusters = st.sidebar.slider("Select Number of Clusters (k)", 2, 10, 4)
model = KMeans(n_clusters=n_clusters, random_state=42)

# ==========================
# Clustering
# ==========================
labels = model.fit_predict(X_pca)

# ==========================
# Evaluation Metrics
# ==========================
st.header("📈 Model Evaluation")

if len(set(labels)) > 1:
    score = silhouette_score(X_pca, labels)
    st.metric("Silhouette Score", f"{score:.3f}")

cluster_counts = pd.Series(labels).value_counts().sort_index()
st.subheader("Cluster Distribution")
st.bar_chart(cluster_counts)

# ==========================
# Cluster vs Purchase History
# ==========================
if 'purchase_history' in df_raw.columns:
    st.header("🛍️ Cluster vs Purchase History")
    crosstab = pd.crosstab(labels, df_raw['purchase_history'],
                           rownames=['Cluster'], colnames=['Purchase History'])
    st.dataframe(crosstab, use_container_width=True)
else:
    st.warning("⚠️ 'purchase_history' column not found in dataset.")

# ==========================
# Cluster Mean Analysis
# ==========================
st.header("📊 Cluster Mean Values")

df_clustered = df_encoded.copy()
df_clustered['Cluster'] = labels
cluster_means = df_clustered.groupby('Cluster').mean()

st.dataframe(cluster_means.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

st.markdown(
    """
**Insights:**

- Customers with **higher purchasing power (around age 50)** typically hold **PhD-level degrees**, are **married**, and demonstrate **strong brand loyalty**.  
  This group represents the company’s **most valuable customers**, making them ideal for **premium offers, exclusive memberships**, or **loyalty programs**.

- The **second group**, although having **lower purchasing power (around age 40)**, also shows **high customer loyalty** and is **mostly married**.  
  They often hold **Master’s or Bachelor’s degrees**, suggesting strong engagement but limited spending capacity.  
  This segment could benefit from **value-based promotions**, **reward programs**, and **personalized deals** to maintain loyalty while encouraging higher spending.

- Overall, **education level and marital status** appear to play a strong role in both **spending behavior** and **brand commitment**.  
  Businesses can use these insights to design **segmented marketing campaigns** and **customer retention strategies**.
"""
)


# ==========================
# PCA Explained Variance
# ==========================
st.header("🧮 PCA Explained Variance")
features = range(len(pca.explained_variance_))
fig, ax = plt.subplots()
ax.bar(features, pca.explained_variance_, color="#69b3a2")
ax.set_xlabel("PCA Feature")
ax.set_ylabel("Variance")
ax.set_xticks(features)
ax.set_title("Variance Explained by Each PCA Component")
st.pyplot(fig)

# ==========================
# Cluster Visualization
# ==========================
st.header("🎯 Cluster Visualization (PCA 2D)")
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Customer Segments in 2D PCA Space")
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="best")
ax.add_artist(legend1)
st.pyplot(fig)

# ==========================
# Sidebar Footer
# ==========================
st.sidebar.markdown("---")
st.sidebar.header("👤 About the Creator")
st.sidebar.markdown(
    """
**Jonathan Wong**  
📚 Data Science  

🔗 [LinkedIn](https://www.linkedin.com/in/jonathan-wong-2b9b39233/)  
🔗 [GitHub](https://github.com/Excitedicecream)
"""
)

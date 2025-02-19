import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Load and Inspect Data
data = pd.read_csv(r"C:\Users\91944\Desktop\Data_Analyst_Project\Bank Customer Churn Prediction.csv")
print("Data loaded successfully")

# Inspect data
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nData information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# 2. Select Relevant Features for Clustering
features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
X = data[features].copy() # Use .copy() to avoid SettingWithCopyWarning

# Handle non-numerical data
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
data['country'] = data['country'].map({'France': 0, 'Spain': 1, 'Germany':2})

X = data[['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary', 'gender', 'country']].copy()

# 3. Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find Optimal Number of Clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# 5. Apply K-Means Clustering
optimal_k = 3  # Choose the optimal K from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualize Customer Segments

# Pair Plots
sns.pairplot(data[features + ['cluster']], hue='cluster')
plt.suptitle('Pair Plots of Customer Segments', y=1.02)
plt.show()

# PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = data['cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis')
plt.title('PCA Visualization of Customer Segments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 7. Analyze High-Risk Churn Groups
cluster_churn_rate = data.groupby('cluster')['churn'].value_counts(normalize=True).unstack()
print("\nChurn rate per cluster:")
print(cluster_churn_rate)

cluster_churn_rate.plot(kind='bar', stacked=True)
plt.title('Churn Rate per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.show()


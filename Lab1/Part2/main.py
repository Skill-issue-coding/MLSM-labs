import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Reading data
SHOPPING_DATA_URL = 'Lab1/Part2/shopping_data.csv'
customer_data = pd.read_csv(SHOPPING_DATA_URL)

# Dropping irrelevant information
# customer_data = customer_data.drop(['CustomerID', 'Genre', 'Age'], axis=1)
data = customer_data.iloc[:, 3:5].values

# print(data)

# Plotting original data
labels = range(1,201)
plt.figure(figsize=(10,7))
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:,0], data[:,1], label='True Position')
plt.show()

# Dendogram
linked = linkage(data, 'ward')
labelList = range(1,201)
plt.figure(figsize=(10,7))
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

# Clustering the points
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')

plt.show()
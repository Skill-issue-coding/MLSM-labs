# Questions

## 1. Can you choose n_components=2? Can you think of some method to test this?

Yes, you absolutely can choose n_components=2. The method to test if this is a good choice is to check the explained variance ratio that is done in code below. The result show that the first two components explain a very large portion of the variance, confirming that using 2 components is a valid and effective choice for visualization and further modeling.

## 2. Create the scatter plot of the third principal component (that is, you combine the third principal component with the first and then the second principal component). What can you see with the plot? What is the difference?

**Plotting the code below shows that:**

The First vs Second component plot showed very clear separation. This is because the first component captures the most variance (information) in the data. The second component captures the next most and the third component captures even less information. Therefore, plots involving the third component will show more overlapping clusters. So the first two components are the most important for distinguishing between the two cancer types. The third component adds little value for this specific classification task.

```python
Xax = X_pca[:, 0]  # First Principal Component
Yax = X_pca[:, 1]  # Second Principal Component
Zax = X_pca[:, 2]  # Third Principal Component
labels = cancer.target
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'}
marker = {0: '*', 1: 'o'}
alpha = {0: .3, 1: .5}

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Plot 1: First vs Third Component
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax1.scatter(Xax[ix], Zax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax1.set_xlabel("First Principal Component", fontsize=14)
ax1.set_ylabel("Third Principal Component", fontsize=14)
ax1.set_title("First vs Third Principal Component")
ax1.legend()

# Plot 2: Second vs Third Component
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax2.scatter(Yax[ix], Zax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax2.set_xlabel("Second Principal Component", fontsize=14)
ax2.set_ylabel("Third Principal Component", fontsize=14)
ax2.set_title("Second vs Third Principal Component")
ax2.legend()

plt.tight_layout()
plt.show()

#  heat-plot to see how the features mixed up to create the components
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()
```

## 3. Can you tell which feature contribute more towards the 1st PC?

By looking at the pca.components_ (heat-plot) above, "mean concave points" and "mean concavity" look to hold significant weight. Printing the features with the highest absolute values in the column contribute the most to the first principal component.

*This is done in code below:*

```python
# Get the loadings for the first principal component (PC1)
pc1_loadings = pca.components_[0]
# Create a list of feature names
feature_names = cancer.feature_names
# Create a DataFrame for easy sorting and visualization
loadings_df = pd.DataFrame({'Feature': feature_names, 'Loading on PC1': pc1_loadings})
# Sort the features by the absolute value of their loading (importance)
# Using absolute value because a large negative value is also very influential.
loadings_df['Abs Loading'] = np.abs(loadings_df['Loading on PC1'])
loadings_df = loadings_df.sort_values('Abs Loading', ascending=False)
# Print the top 10 features contributing to PC1
print("Top 10 features contributing to the First Principal Component:")
print(loadings_df[['Feature', 'Loading on PC1']].head(10))


# show the correlation plot of ‘worst’ values of the features
cancer_df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
feature_worst = list(cancer_df.columns[20:31]) # select the 'worst' features
import seaborn as sns
s=sns.heatmap(cancer_df[feature_worst].corr(),cmap='coolwarm')
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()
```

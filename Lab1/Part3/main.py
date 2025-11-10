import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#  importing the cancer dataset.
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# basic description of the data-set
# print(cancer.DESCR)

# just to make sure 0 represents malignant
# print(len(cancer.data[cancer.target==1]))

# ----- Histograms of malignant and benign classes -----
'''
# If the two histograms are separated based on the feature => the feature is important to discern the instances

# 3 columns each containing 10 figures, total 30 features
fig,axes = plt.subplots(10,3, figsize=(12, 9))
malignant = cancer.data[cancer.target==0] # define malignant
benign = cancer.data[cancer.target==1] # define benign
ax = axes.ravel() # flat axes with numpy ravel
for i in range(30):
 _,bins=np.histogram(cancer.data[:,i],bins=40)
 ax[i].hist(malignant[:,i],bins=bins,color='r',alpha=.5) # red color for malignant class
 ax[i].hist(benign[:,i],bins=bins,color='g',alpha=0.3 ) # alpha is for transparency in the
 # overlapped region
 ax[i].set_title(cancer.feature_names[i],fontsize=9)
 # the x-axis coordinates are not so useful as we just want to look how well separated
 # the histograms are
 ax[i].axes.get_xaxis().set_visible(False)
 ax[i].set_yticks(())
ax[0].legend(['malignant','benign'],loc='best',fontsize=8)
plt.tight_layout() # let's make good plots
plt.show()
'''

# ----- Scatter plots with few features of cancer dataset -----
'''
# just convert the scikit learn data-set to pandas data-frame.
cancer_df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
plt.subplot(1,2,1) # first plot
plt.scatter(cancer_df['worst symmetry'], cancer_df['worst texture'],
    s = cancer_df['worst area']*0.05, color='magenta', label='check',alpha=0.3)
plt.xlabel('Worst Symmetry',fontsize=12)
plt.ylabel('Worst Texture',fontsize=12)
plt.subplot(1,2,2) # 2nd plot
plt.scatter(cancer_df['mean radius'], cancer_df['mean concave points'],
    s = cancer_df['mean area']*0.05, color='purple',label='check', alpha=0.3)
plt.xlabel('Mean Radius',fontsize=12)
plt.ylabel('Mean Concave Points',fontsize=12)
plt.tight_layout()
plt.show()
'''

# ----- PCA -----

# first scale the data such that each feature has unit variance
scaler = StandardScaler() #instantiate
# compute the mean and standard which will be used in the next command
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
# we check the minimum and maximum of the scaled features which we expect to be 0 and 1
print("after scaling minimum", X_scaled.min(axis=0))

# apply PCA on the scaled dataset
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("shape of X_pca", X_pca.shape) # let's check the shape of X_pca array

ex_variance = np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)

# Visualizing based on the first two principal components
Xax = X_pca[:,0]
Yax = X_pca[:,1]
labels = cancer.target
cdict = {0:'red',1:'green'}
labl = {0:'Malignant',1:'Benign'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}
fig,ax = plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix = np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

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


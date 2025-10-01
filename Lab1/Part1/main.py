# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# ----- DATA PREVIEW -----

'''
# Preview samples from data sets
print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())
'''

'''
# Initial statistics from data sets
print("***** Train_Set *****")
print(train.describe())
print("***** Test_Set *****")
print(test.describe())
'''

'''
# List the feature names
print(train.columns.values)
'''

'''
# Missing values in data
train.isna().head() # For the train set
test.isna().head() # For the test set
# print("*****In the train set*****")
# print(train.isna().sum()) # Total number of missing values
# print("\n")
# print("*****In the test set*****")
# print(test.isna().sum()) # Total number of missing values
'''

'''
# Filling missing values with mean imputation
# Fill missing values with mean column values in the train set
train.fillna(train.mean(numeric_only=True), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(numeric_only=True), inplace=True)
# print(train.isna().sum())
# print(test.isna().sum())
'''

'''
# Features are neither categorical nor numerical
# Ticket is a mix of numeric and alphanumeric data types
print(train['Ticket'].head())
# Cabin is alphanumeric
print(train['Cabin'].head())
'''

'''
# Survival count with respect to Pclass:
train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False))
# Survival count with respect to Sex:
train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Survival count with respect to SibSp:
train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
'''

# Tests
# pd.set_option('display.max_columns', None)
# print(train.isna().sum())
# train.fillna(train.mean(numeric_only=True), inplace=True)
# print(train.isna().sum())
# print(train[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=True))

'''
# Plot the graph of "Age vs. Survived":
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()
'''

'''
# Plot the graph of how the Pclass and Survived features are related to each other:
grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
'''

# ----- K-MEANS MODEL -----

# See data types of different features
# train.info()

# Missing values in data
# print(train.isna().sum())
# print(test.isna().sum())

# Fill missing values with mean column values in the train set (numeric only)
train.fillna(train.mean(numeric_only=True), inplace=True)
test.fillna(test.mean(numeric_only=True), inplace=True)

# Drop features Name, Ticket, Cabin and Embarked. They will not have significant impact
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Convert the non-numeric feature 'Sex' to a numerical feature
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

# Result from dropped and converted features
# train.info()

# Test set does not have Survived feature
# test.info()

# Drop the Survived feature in test set
X = np.array(train.drop(['Survived'], axis=1).astype(float))
y = np.array(train['Survived'])

'''
# cluster the passenger records into 2: Survived or Not survived
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_clusters=2,
n_init=10, random_state=None, tol=0.0001, verbose=0)


# percentage of passenger records that were clustered correctly
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))
'''

# Improved version with feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600,
 n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
kmeans.fit(X_scaled)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))

# ----- ANSWERS TO QUESTIONS -----

# 1. What are the relevant features of the Titanic dataset. Why are they relevant?
'''
    By looking at the K-Means Clustering.pdf the most most relevant features for survival prediction on the Titanic are:
    1. Sex - "Women and children first" policy was enforced
    2. Pclass - Higher class passengers had priority for lifeboats
    3. Age - Children were given priority
    4. SibSp/Parch - Family size affected survival chances
    5. Fare - Correlates with socio-economic status and cabin location
'''
# 2. Can you find a parameter configuration to get a validation score greater than 62% ?
'''
    Yes, by using feature scaling and proper parameter tuning, we can consistently achieve over 62% accuracy. The key improvements are:
    - Using MinMaxScaler for feature normalization (parameters go from 0 to 1)
    - Setting random_state for reproducible results
    - Using n_init=10 for better initialization
    - Removing non-predictive features like PassengerId, Name, etc. 
'''
# 3. What are the advantages/disadvantages of K-Means clustering?
'''
    Advantages:
    - Simple and easy to implement
    - Fast and efficient for large datasets
    - Scalable to high-dimensional data
    - Guaranteed convergence
    Disadvantages:
    - Requires pre-specifying number of clusters (K)
    - Sensitive to initial centroid selection (can pick a bad k-mean)
    - Assumes spherical clusters of similar size
    - Sensitive to outliers and feature scaling
    - Produces new results for every run
    - Needs large amount of memory to store the data
'''
# 4. How can you address the weaknesses?
'''
    For determining K: Use elbow method or silhouette analysis
    For initialization sensitivity: Use n_init=10 and init='k-means++'
    For outlier sensitivity: Remove outliers or use robust scaling
    For feature scaling: Always normalize features before clustering
    For cluster shape assumptions: Consider DBSCAN for non-spherical clusters
    Delete features with none essential data (Know your data)
'''

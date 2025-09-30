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

# train.info()

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


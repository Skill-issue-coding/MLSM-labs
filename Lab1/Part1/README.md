# Questions

## 1. What are the relevant features of the Titanic dataset. Why are they relevant?

By looking at the K-Means Clustering.pdf the most most relevant features for survival prediction on the Titanic are:

1. Sex - "Women and children first" policy was enforced

2. Pclass - Higher class passengers had priority for lifeboats

3. Age - Children were given priority

4. SibSp/Parch - Family size affected survival chances

5. Fare - Correlates with socio-economic status and cabin location

## 2. Can you find a parameter configuration to get a validation score greater than 62% ?

Yes, by using feature scaling and proper parameter tuning, we can consistently achieve over 62% accuracy. 

**The key improvements are:**

- Using MinMaxScaler for feature normalization (parameters go from 0 to 1)

- Setting random_state for reproducible results

- Using n_init=10 for better initialization

- Removing non-predictive features like PassengerId, Name, etc.

## 3. What are the advantages/disadvantages of K-Means clustering?

**Advantages:**

- Simple and easy to implement

- Fast and efficient for large datasets

- Scalable to high-dimensional data

- Guaranteed convergence

**Disadvantages:**

- Requires pre-specifying number of clusters (K)

- Sensitive to initial centroid selection (can pick a bad k-mean)

- Assumes spherical clusters of similar size

- Sensitive to outliers and feature scaling

- Produces new results for every run

- Needs large amount of memory to store the data

## 4. How can you address the weaknesses?

**For determining K:** Use elbow method or silhouette analysis

**For initialization sensitivity:** Use n_init=10 and init='k-means++'

**For outlier sensitivity:** Remove outliers or use robust scaling

**For feature scaling:** Always normalize features before clustering

**For cluster shape assumptions:** Consider DBSCAN for non-spherical clusters

Delete features with none essential data (Know your data)

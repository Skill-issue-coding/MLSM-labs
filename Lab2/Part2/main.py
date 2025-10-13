# Building a KNN Classifier in Scikit-learn

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn import metrics

'''
""" Defining dataset """
# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
# Label or target variable
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


""" Encoding data columns """
# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers
weather_encoded=le.fit_transform(weather)
print(weather_encoded)

temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)


""" Combining Features """
features=list(zip(weather_encoded,temp_encoded))


""" Generating the Model """
model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)
# Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Prediction: " ,predicted)
'''

""" KNN with Multiple Labels """
""" Loading Data """
# Load dataset
wine = datasets.load_wine()


""" Exploring Data 
# print the names of the features
print(wine.feature_names)

# print the label species(class_0, class_1, class_2)
print(wine.target_names)

# print the wine data (top 5 records)
print(wine.data[0:5])

# print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)
print(wine.target)

# print data(feature)shape
print(wine.data.shape)

# print target(or label)shape
print(wine.target.shape)
"""


""" Splitting Data """
# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)


""" Generating the Model for K=5 """
# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
knn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = knn.predict(X_test)


""" Model Evaluation for k=5 """
# Model Accuracy, how often is the classifier correct?
print("Accuracy for k=5: ",metrics.accuracy_score(y_test, y_pred))


""" Regenerating Model for K=7 """
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


""" Model Evaluation for k=7 """
print("Accuracy for k=7: ",metrics.accuracy_score(y_test, y_pred))

""" QUESTIONS
1. Why choosing a good value for k is important in KNN?

    The value of k is the most critical hyperparameter in KNN because it directly controls the trade-off between,
    the model's bias and variance, which determines its ability to generalize to new, unseen data.
    If k is too small the model becomes very complex and has low bias but high variance.
    It will capture noise and outliers in the training data, leading to overfitting. The decision boundary becomes very jagged.
    With k=1, a new point is classified based on its single nearest neighbor, which might be an anomalous data point.
    
    If k is too large the model becomes very simple and has high bias but low variance.
    It will oversimplify the model and fail to capture important patterns, leading to underfitting. 
    The decision boundary becomes overly smooth.
    With a very large k, the prediction will always be simply the majority class in the entire dataset, ignoring local patterns.
    
2. How can you decide a good value for k?

    Using cross-validation on the training set. Define a range of possible k values (e.g., from 1 to 20).
    For each value of k in that range, perform k-fold cross-validation. 
    Calculate the average performance metric across all folds for that k.
    Plot the average performance against the k values.
    Select the k that gives the highest cross-validation performance. 
    Often, you look for the simplest model (larger k) that performs just as well as more complex ones, 
    which is usually found at the "elbow" of the curve where performance starts to plateau or degrade.
    
3. Can you use KNN to classify non-linearly separable data?

    Yes, it creates a highly flexible, piecewise linear decision boundary that can adapt to very complex, non-linear shapes. 
    The boundary is formed by the Voronoi tessellation of the feature space. 
    This makes KNN very powerful for datasets where the classes cannot be separated by a simple straight line.
    
4. Is KNN sensible to the number of features in the dataset?

    Yes, KNN is highly sensitive to the number of features. As the number of features (dimensions) increases, 
    the volume of the feature space grows exponentially. This causes the training data to become increasingly sparse.
    In high-dimensional space, the concept of "nearest neighbors" becomes less meaningful,
    because the distance between any two points converges to be the same.
    It becomes much harder to find truly informative neighbors, and the model's performance can degrade significantly.
    This also makes KNN computationally expensive.
    
5. Can you use KNN for a regression problem?

    Yes. The algorithm is called K-Nearest Neighbors Regression. Instead of taking a majority vote for classification, 
    the prediction for a new data point is the average (or sometimes weighted average) of the target values of its k nearest neighbors.
    
6. What are the Pros and Cons of KNN?

    Pros:
    It's much faster compared to other classification algorithms.
    No Training Phase so it's a "lazy learner." The model simply stores the training data, making the training step very fast.
    KNN can be useful in case of nonlinear data.
    Highly Effective with Large Enough Datasets and can model very complex, non-linear decision boundaries,
    without assuming any underlying data distribution.
    
    Cons:
    The testing phase of KNN is slower and costlier in terms of time and memory
    Computationally Expensive Prediction as the "lazy" nature means all the computation is deferred to the prediction time. 
    Finding the nearest neighbors in a large dataset can be very slow.
    Memory Intensive and requires storing the entire training dataset.
    Sensitive to the Curse of Dimensionality. The Euclidean distance is sensitive to magnitudes; 
    the features with high magnitudes will weight more than the features with low magnitudes

"""
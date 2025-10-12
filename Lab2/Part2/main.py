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
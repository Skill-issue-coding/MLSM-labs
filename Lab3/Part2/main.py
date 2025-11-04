"""
==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_random_state

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the faces datasets
data, targets = fetch_olivetti_faces(return_X_y=True)

train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces,))
test = test[face_ids, :]

n_pixels = data.shape[1]
# Upper half of the faces
X_train = train[:, : (n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2 :]
X_test = test[:, : (n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2 :]

# Fit estimators
ESTIMATORS = {
    #"Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
    #"K-nn": KNeighborsRegressor(),
    #"Linear reg.": LinearRegression(),
    #"Ridge": RidgeCV(),
    # 3.
    # a.
    #"RDT(10,50)": DecisionTreeRegressor(max_depth=10, max_features=50 ,random_state=0),
    # b.
    #"RDT(20,50)": DecisionTreeRegressor(max_depth=20, max_features=50 ,random_state=0),
    # c.
    #"RDT(20,25)": DecisionTreeRegressor(max_depth=20, max_features=25 ,random_state=0),
    # d.
    #"RF(10,50)": RandomForestRegressor(n_estimators=10, max_depth=10, max_features=50, random_state=0),
    # e.
    #"RF(20,50)": RandomForestRegressor(n_estimators=10, max_depth=20, max_features=50, random_state=0),
    # f.
    #"RF(20,25)": RandomForestRegressor(n_estimators=10, max_depth=20, max_features=25, random_state=0),
    # 4.
    "RF Test 1": RandomForestRegressor(n_estimators=250, max_depth=20, max_features=25, random_state=0, min_samples_split=5, min_samples_leaf=2),
    "RF Test 2": RandomForestRegressor(n_estimators=250, max_depth=20, max_features="sqrt", random_state=0, min_samples_split=5, min_samples_leaf=2),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

    sub.axis("off")
    sub.imshow(
        true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest"
    )

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

        sub.axis("off")
        sub.imshow(
            completed_face.reshape(image_shape),
            cmap=plt.cm.gray,
            interpolation="nearest",
        )

plt.show()

"""

1.  Explain what the program does.

    This program takes the top half of face images as input and uses machine learning to predict the bottom half. 
    It trains four different models to complete missing facial features, 
    then compares how well each method reconstructs the full face on people it has never seen before. 
    The goal is to see which algorithm best understands general facial structure to fill in missing parts.

2.  What is your interpretation of the final plot? Which algorithm has better performance
    in building the unknown parts of the face?
    
    based of our interpretation the method using extra trees is the overall most accurate, but sometimes
    the other methods can produce a more satisfying result. 
    
3.  Download the code from the link above and modify it by adding the results of the following
    algorithms to the final plot:
    (a) Regression decision tree with max depth of 10 and max number of features 50
    (b) Regression decision tree with max depth of 20 and max number of features 50
    (c) Regression decision tree with max depth of 20 and max number of features 25
    (d) Random forest with max depth of 10 and max number of features 50
    (e) Random forest with max depth of 20 and max number of features 50
    (f) Random forest with max depth of 20 and max number of features 25
    How do you interpret the results?
    
    When looking at the result we see that regression decision tree has a clearer prediction (less noise). 
    Extra trees is still the best predictor, but sometimes the random forest outperforms it. For example on
    on the last person. Somehow regression decision tree can sometimes change the ethnicity of a person
    add facial hair. 
    
4.  How could performance of random forest be improved? (Hint: have a look at the
    example of using Haar-like feature in face detection here: https://realpython.com/
    traditional-face-detection-python/)
    
    Random Forest performance can be improved by:

    1. Better Features: 
        Create smarter features (like Haar features do for faces) instead of just using raw data

    2. Feature Selection: 
        Keep only the most important features to reduce noise

    3. Parameter Tuning: 
        Optimize tree depth, number of trees, and split criteria

    4. Data Preparation: 
        Handle imbalanced data and scale features properly
    
    5. Model Combinations: 
        Use multiple forests together or with other algorithms

    Key Insight: 
    Like Haar features show, intelligent feature design often matters more than complex models. 
    Focus on creating better features rather than just making the model bigger.

"""
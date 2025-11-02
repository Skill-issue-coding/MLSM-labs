""" Validation Metrics 
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
"""

""" Regression 
#from sklearn.datasets import load_boston
#boston = load_boston()
X=data
Y=target
cv = 10

print('\nlinear regression')
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nridge regression')
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlasso regression')
lasso = Lasso(alpha=0.1)
scores = cross_val_score(lasso, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\ndecision tree regression')
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nrandom forest regression')
forest = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(forest, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlinear support vector machine')
svm_lin = svm.SVR(epsilon=0.2,kernel='linear',C=1)
scores = cross_val_score(svm_lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nsupport vector machine rbf')
clf = svm.SVR(epsilon=0.2,kernel='rbf',C=1.)
scores = cross_val_score(clf, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nknn')
knn = KNeighborsRegressor()
scores = cross_val_score(knn, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(knn, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
"""

""" Recursive Feature Elimination (RFE) 

from sklearn.feature_selection import RFE
best_features=4

#rfe_lin = RFE(lin,best_features).fit(X,Y)
#supported_features=rfe_lin.get_support(indices=True)
#for i in range(0, 4):
#    z=supported_features[i]
#    print(i+1,data.feature_names[z])

print("\nLinear Regression with RFE")
rfe_lin = RFE(lin,step=best_features).fit(X, Y)
mask = np.array(rfe_lin.support_)
score = cross_val_score(lin, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lin, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRidge Regression with RFE")
rfe_ridge = RFE(ridge,step=best_features).fit(X, Y)
mask = np.array(rfe_ridge.support_)
score = cross_val_score(ridge, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(ridge, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLasso Regression with RFE")
rfe_lasso = RFE(lasso,step=best_features).fit(X, Y)
mask = np.array(rfe_lasso.support_)
score = cross_val_score(lasso, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lasso, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nDecision Tree regression with RFE")
rfe_tree = RFE(tree, step=best_features).fit(X, Y)
mask = np.array(rfe_tree.support_)
score = cross_val_score(tree, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(tree, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRandom Forest regression with RFE")
rfe_forest = RFE(forest, step=best_features).fit(X, Y)
mask = np.array(rfe_forest.support_)
score = cross_val_score(forest, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(forest, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLinear Support Vector Machine regression with RFE")
rfe_svm_lin = RFE(svm_lin, step=best_features).fit(X, Y)
mask = np.array(rfe_svm_lin.support_)
score = cross_val_score(svm_lin, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(svm_lin, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))
"""

""" Car Evaluation Quality """
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('data_cars.csv', header=None)
for i in range(len(df.columns)):
    df[i] = df[i].astype('category')
print(df.head())

# map categories to values
map0 = dict(zip(df[0].cat.categories, range(len(df[0].cat.categories))))
map1 = dict(zip(df[1].cat.categories, range(len(df[1].cat.categories))))
map2 = dict(zip(df[2].cat.categories, range(len(df[2].cat.categories))))
map3 = dict(zip(df[3].cat.categories, range(len(df[3].cat.categories))))
map4 = dict(zip(df[4].cat.categories, range(len(df[4].cat.categories))))
map5 = dict(zip(df[5].cat.categories, range(len(df[5].cat.categories))))
map6 = dict(zip(df[6].cat.categories, range(len(df[6].cat.categories))))

cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
df = df.iloc[np.random.permutation(len(df))]
print(df.head())

df_f1 = pd.DataFrame(columns=['method'] + sorted(map6, key=map6.get))
df_precision = pd.DataFrame(columns=['method'] + sorted(map6, key=map6.get))
df_recall = pd.DataFrame(columns=['method'] + sorted(map6, key=map6.get))


def CalcMeasures(method, y_pred, y_true, df_f1=df_f1, df_precision=df_precision, df_recall=df_recall):
    df_f1.loc[len(df_f1)] = [method] + list(f1_score(y_true, y_pred, average=None, zero_division=0))
    df_precision.loc[len(df_precision)] = [method] + list(precision_score(y_true, y_pred, average=None, zero_division=0))
    df_recall.loc[len(df_recall)] = [method] + list(recall_score(y_true, y_pred, average=None, zero_division=0))


X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

cv = 10
method = 'linear support vector machine'
clf = svm.SVC(kernel='linear', C=50)
y_pred = cross_val_predict(clf, X, Y, cv=cv)
CalcMeasures(method, y_pred, Y)

method = 'naive bayes'
clf = MultinomialNB()
y_pred = cross_val_predict(clf, X, Y, cv=cv)
CalcMeasures(method, y_pred, Y)

method = 'logistic regression'
clf = LogisticRegression()
y_pred = cross_val_predict(clf, X, Y, cv=cv)
CalcMeasures(method, y_pred, Y)

method = 'k nearest neighbours'
clf = KNeighborsClassifier(weights='distance', n_neighbors=5)
y_pred = cross_val_predict(clf, X, Y, cv=cv)
CalcMeasures(method, y_pred, Y)

print("\nf-measure:")
print(df_f1)
print("\nPrecision:")
print(df_precision)
print("\nRecall:")
print(df_recall)

labels_counts = df[6].value_counts()
print("\nNumber of samples in each class:")
print(pd.Series(map6).map(labels_counts))
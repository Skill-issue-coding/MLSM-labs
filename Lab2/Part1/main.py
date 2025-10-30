import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge

""" Simple Linear Regression
# scattered data
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)
plt.show()

# fit data and construct the best-fit line
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

# slope and intercept of the data
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)

# multidimensional regressions
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)
print(model.intercept_)
print(model.coef_)
"""


""" Polynomial basis functions 
# polynomial projection
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])
print(poly.fit_transform(x[:, None]))

# use a pipeline
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())

# use the linear model to fit more complex relationships between x and y (sin wave with noise)
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
xfit = np.linspace(0, 10, 1000)
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()


''' Gaussian basis functions '''

# custom transformer that will create Gaussian basis functions
class GaussianFeatures(BaseEstimator, TransformerMixin):
    '''Uniformly spaced Gaussian features for one-dimensional input'''

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)

gauss_model = make_pipeline(GaussianFeatures(20),LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)
plt.show()


''' Regularization '''
# overfitting
model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.show()

# reason for overfit
def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    if title:
        ax[0].set_title(title)
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))
    plt.show()
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)


''' Ridge regression (L2 Regularization) '''
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')


''' Lasso regression (L1 regularization) '''
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')
"""


""" Case Study: Predicting Bicycle Traffic (OPTIONAL PART) """


""" QUESTIONS 
1. When can you use linear regression?
    
    You can use linear regression when you expect or observe a linear relationship between variables
    — that is, when the dependent variable can be modeled as a weighted sum of one or more independent
    variables.
    It works well as a starting point for regression tasks because it’s fast to fit, easy to interpret, 
    and provides a baseline model.
    
2. How can you generalize linear regression models to account for more complex relationships among the data?
    
    We can generalize linear regression by using basis functions. 
    The core linear model y = w₀ + w₁x is limited to straight lines. 
    By replacing the input variable x with a transformed version, φ(x),
    we can fit a linear model in the new feature space that is non-linear in the original input space.
    The model becomes: y = w₀ + w₁φ₁(x) + w₂φ₂(x) + ... + wₘφₘ(x).
    This is still "linear" because it's linear with respect to the weights w, 
    but the function φ(x) can be non-linear, allowing us to capture curves, waves, and other complex
    patterns, simply by projected them into a higher dimension.
    
3. What are the basis functions?

    They are fixed, non-linear functions that we apply to our input data x to project it into a 
    higher-dimensional space.
    In this lab we used Polynomial Basis Functions and Gaussian Basis Functions.
    
4. How many basis functions can you use in the same regression model?

    There is no limit. However, the number is a hyperparameter that must choose, and it has a critical trade-off.
    Too few and the model will be too simple and underfit the data, failing to capture the underlying trend (high bias).
    Too many and the model will become too complex and overfit the data. 
    It will memorize the noise in the training set instead of learning the generalizable pattern, 
    leading to poor performance on new data (high variance).
    
5. Can overfitting be a problem? And if so, what can you do about it?

    Yes, overfitting is a major problem, especially when using flexible models with many basis functions.
    Regularization is the primary weapon against overfitting in linear models. 
    It works by adding a penalty to the model's loss function that discourages the weights from becoming too large.
    Ridge Regression adds a penalty proportional to the square of the weights. (good when all variables contribute 
    to the outcome but you want to Reduce their overall impact)
    This shrinks weights but rarely sets them to exactly zero. It's very effective at controlling model complexity.
    Lasso Regression adds a penalty proportional to the absolute value of the weights. (Good when some variables are unessessary)
    This can drive some weights to exactly zero, effectively performing feature selection and creating a sparser model.
    
"""
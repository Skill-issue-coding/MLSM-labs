# Questions

## 1. When can you use linear regression?

You can use linear regression when you expect or observe a linear relationship between variables — that is, when the dependent variable can be modeled as a weighted sum of one or more independent variables. It works well as a starting point for regression tasks because it’s fast to fit, easy to interpret, and provides a baseline model.

## 2. How can you generalize linear regression models to account for more complex relationships among the data?

We can generalize linear regression by using basis functions. The core linear model y = w₀ + w₁x is limited to straight lines. By replacing the input variable x with a transformed version, φ(x), we can fit a linear model in the new feature space that is non-linear in the original input space. The model becomes: y = w₀ + w₁φ₁(x) + w₂φ₂(x) + ... + wₘφₘ(x). This is still "linear" because it's linear with respect to the weights w, but the function φ(x) can be non-linear, allowing us to capture curves, waves, and other complex patterns, simply by projected them into a higher dimension.

## 3. What are the basis functions?

They are fixed, non-linear functions that we apply to our input data x to project it into a higher-dimensional space. In this lab we used Polynomial Basis Functions and Gaussian Basis Functions.

## 4. How many basis functions can you use in the same regression model?

There is no limit. However, the number is a hyperparameter that must be chosen, and it has a critical trade-off. Too few and the model will be too simple and underfit the data, failing to capture the underlying trend (high bias). Too many and the model will become too complex and overfit the data. It will memorize the noise in the training set instead of learning the generalizable pattern, leading to poor performance on new data (high variance).

## 5. Can overfitting be a problem? And if so, what can you do about it?

Yes, overfitting is a major problem, especially when using flexible models with many basis functions. Regularization is the primary weapon against overfitting in linear models. It works by adding a penalty to the model's loss function that discourages the weights from becoming too large. Ridge Regression adds a penalty proportional to the square of the weights. (good when all variables contribute to the outcome but you want to Reduce their overall impact) This shrinks weights but rarely sets them to exactly zero. It's very effective at controlling model complexity. Lasso Regression adds a penalty proportional to the absolute value of the weights. (Good when some variables are unessessary) This can drive some weights to exactly zero, effectively performing feature selection and creating a sparser model.

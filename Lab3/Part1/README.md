# Gaussian NB algorithm

Predict which variant of Iris the flower is.

Features are:

1) Sepal length
2) Sepal width
3) Petal length
4) Petal width

## Code structure
1) Split data into train and test.
2) Structure the train data, make an array for each variant of Iris.
3) Then calculate prior probability.
4) After that calculate the mean and standard deviation for each feature. 
5) Next step is to get the class with the largest posterior probability.

(Posterior probability is calculated by joint_prob / marginal_prob for each class.
Joint probability is prior * likelihood, likelihood is all normal distributions multiplied together.
marginal probability sums all classes joint probability)

Gaussian Naive Bayes works by learning the normal distribution of each feature for every class during training.
When predicting, it calculates how likely each feature value is for each class using the bell curve,
multiplies these probabilities together (assuming feature independence),
and chooses the class with the highest resulting probability.

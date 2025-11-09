# Gaussian NB algorithm

Gaussian Naive Bayes works by learning the normal distribution of each feature for every class during training.
When predicting, it calculates how likely each feature value is for each class using the bell curve,
multiplies these probabilities together (assuming feature independence),
and chooses the class with the highest resulting probability.

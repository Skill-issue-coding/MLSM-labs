# Part1

# Implementation of a Gaussian Naive Bayes classifier from scratch

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics
from math import pi
from math import e


class GaussNB:
    # Class-level attributes to store statistics and target labels
    summaries = {}
    target_values = []

    def __init__(self):
        pass

    def group_by_class(self, data, target):
        """
        :param data: Training set
        :param target: the list of class labels labeling data
        :return:
        Separate the data by their target class;
        that is, create one group for every value of the target class.
        """
        # Create a list of arrays, one for each class in target_values
        separated = [[x for x, t in zip(data, target) if t == c] for c in self.target_values]
        # Convert lists to numpy arrays
        groups = [np.array(separated[0]), np.array(separated[1]), np.array(separated[2])]
        return groups

    def summarize(self, data):
        """
        :param data: a dataset whose rows are arrays of features
        :return:
        the mean and the standard deviation for each feature of data.
        """
        # Iterate over columns (features)
        for index in range(data.shape[1]):
            feature_column = data.T[index]  # Extract a single feature column
            yield {'stdev': statistics.stdev(feature_column), 'mean': statistics.mean(feature_column)}

    def train(self, data, target):
        """
        :param data: a dataset
        :param target: the list of class labels labeling data
        :return:
        For each target class:
            1. Compute prior probability of the class (P(class))
            2. Compute summary statistics (mean and stdev) for every feature
        """
        groups = self.group_by_class(data, target)
        for index in range(len(groups)):
            group = groups[index]
            # Store prior probability and feature summaries per class
            self.summaries[self.target_values[index]] = {
                'prior_prob': len(group) / len(data),
                'summary': list(self.summarize(group))
            }

    def normal_pdf(self, x, mean, stdev):
        """
        :param x: the value of a feature
        :param mean: μ - mean of the feature
        :param stdev: σ - standard deviation of the feature
        :return: Gaussian (Normal) Probability Density function
        N(x; μ, σ) = (1 / (√(2π)σ)) * e^(-(x−μ)² / (2σ²))
        """
        variance = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = -exp_squared_diff / (2 * variance)
        exponent = e ** exp_power
        denominator = ((2 * pi) ** 0.5) * stdev
        normal_prob = exponent / denominator
        return normal_prob

    def marginal_pdf(self, joint_probabilities):
        """
        :param joint_probabilities: dictionary of joint probabilities for each class
        :return:
        Marginal Probability Density Function (normalizing constant).
        It's the sum of all joint probabilities over all classes.
        """
        marginal_prob = sum(joint_probabilities.values())
        return marginal_prob

    def joint_probabilities(self, data):
        """
        :param data: a single observation (list or array of feature values)
        :return:
        Compute the joint probability P(class) * Π P(x_i | class)
        for each class in target_values.
        """
        joint_probs = {}
        for y in range(self.target_values.shape[0]):
            target_v = self.target_values[y]
            item = self.summaries[target_v]
            total_features = len(item['summary'])
            likelihood = 1
            # Compute product of feature likelihoods given the class
            for index in range(total_features):
                feature = data[index]
                mean = self.summaries[target_v]['summary'][index]['mean']
                stdev = self.summaries[target_v]['summary'][index]['stdev'] ** 2  # Note: squared here
                normal_prob = self.normal_pdf(feature, mean, stdev)
                likelihood *= normal_prob
            prior_prob = self.summaries[target_v]['prior_prob']
            # Joint probability = prior * likelihood
            joint_probs[target_v] = prior_prob * likelihood
        return joint_probs

    def posterior_probabilities(self, test_row):
        """
        :param test_row: single observation to classify
        :return:
        Compute the posterior probability P(class | features)
        for each possible class using Bayes' theorem.
        """
        posterior_probs = {}
        joint_probabilities = self.joint_probabilities(test_row)
        marginal_prob = self.marginal_pdf(joint_probabilities)
        # Compute posterior = joint / marginal for each class
        for y in range(self.target_values.shape[0]):
            target_v = self.target_values[y]
            joint_prob = joint_probabilities[target_v]
            posterior_probs[target_v] = joint_prob / marginal_prob
        return posterior_probs

    def get_map(self, test_row):
        """
        :param test_row: single observation to classify
        :return:
        Return the class with the largest posterior probability (MAP estimate)
        """
        posterior_probs = self.posterior_probabilities(test_row)
        target = max(posterior_probs, key=posterior_probs.get)
        return target

    def predict(self, data):
        """
        :param data: test dataset
        :return:
        Predict the most likely class label for each observation.
        """
        predicted_targets = []
        for row in data:
            predicted = self.get_map(row)
            predicted_targets.append(predicted)
        return predicted_targets

    def accuracy(self, ground_true, predicted):
        """
        :param ground_true: list of actual (true) class labels
        :param predicted: list of predicted class labels
        :return:
        Compute the classifier’s accuracy as the proportion of correct predictions.
        """
        correct = 0
        for x, y in zip(ground_true, predicted):
            if x == y:
                correct += 1
        return correct / ground_true.shape[0]


def main():
    # Instantiate custom Gaussian Naive Bayes classifier
    nb = GaussNB()

    # Load the Iris dataset from sklearn
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    # Store the unique class labels
    nb.target_values = np.unique(target)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    # Train the classifier
    nb.train(X_train, y_train)

    # Predict class labels for the test set
    predicted = nb.predict(X_test)

    # Compute model accuracy
    accuracy = nb.accuracy(y_test, predicted)
    print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    main()

"""
Gaussian Naive Bayes works by learning the normal distribution of each feature for every class during training. 
When predicting, it calculates how likely each feature value is for each class using the bell curve, 
multiplies these probabilities together (assuming feature independence), 
and chooses the class with the highest resulting probability.
"""
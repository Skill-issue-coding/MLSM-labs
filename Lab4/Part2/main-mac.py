from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the data
# Could be: 'Lab4/Part2/data/movie_reviews' or 'data/movie_reviews'
MOVIE_DIR: str = 'data/movie_reviews'
movie = load_files(MOVIE_DIR, shuffle=True)

# Testing / Printing data
print(len(movie.data))
print(movie.target_names)
print(movie.data[0][:500])
print(movie.filenames[0])
print(movie.target[0])

# Split data into training and test sets
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size=0.20, random_state=12)

# initialize CountVectorizer
movieVzer = CountVectorizer(min_df=2, max_features=3000, stop_words='english') # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and transform using training text
docs_train_counts = movieVzer.fit_transform(docs_train)

# Testing / printing word indices
print(movieVzer.vocabulary_.get('screen')) # 'screen' is found in the corpus, mapped to index 2307
print(movieVzer.vocabulary_.get('seagal')) # Likewise, Mr. Steven Seagal is present... (2314)
print(docs_train_counts.shape) # huge dimensions! 1,600 documents, 3K unique terms.

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
print(docs_train_tfidf.shape)

# Using the fitted vectorizer and transformer, transform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
print(metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) # [[TP, FP], [FN, TN]]

""" Trying the classifier on fake movie reviews """
# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
            'Steven Seagal was terrible', 'Steven Seagal shone through.',
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

""" Pipeline """

# Naive Bayes pipeline
text_clf_nb = Pipeline([
 ('vect', CountVectorizer(min_df=2, max_features=3000, stop_words='english')),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

# train the model
text_clf_nb.fit(docs_train, y_train)

# Evaluation of the performance on the test set
predicted_nb = text_clf_nb.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted_nb == y_test))

# SVM pipeline
text_clf_svm = Pipeline([
 ('vect', CountVectorizer(min_df=2, max_features=3000, stop_words='english')),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])

# Evaluation of the performance on the test set
text_clf_svm.fit(docs_train, y_train)
predicted_svm = text_clf_svm.predict(docs_test)
print("SVM accuracy ",np.mean(predicted_svm == y_test))

# more detailed performance analysis of the results
print("\nNaive Bayes Classification Report:")
print(metrics.classification_report(y_test, predicted_nb, target_names=movie.target_names))

print("\nSVM Classification Report:")
print(metrics.classification_report(y_test, predicted_svm, target_names=movie.target_names))

# confusion matrix
# print("\nNaive Bayes Confusion Matrix:")
# print(metrics.confusion_matrix(movie.target, predicted_nb)) # [[TP, FP], [FN, TN]]

print("\nSVM Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted_svm)) # [[TP, FP], [FN, TN]]

""" Grid Search """

# Fresh pipeline for grid search
grid_pipeline = Pipeline([
    ('vect', CountVectorizer(min_df=2, max_features=3000, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=42)),
])

# parameters for grid search
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2), (1,3),(2,3)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3, 1e-4),
 'clf__max_iter': [1000, 2000]
}

# parallelize search on training data on multiple CPU cores
gs_clf = GridSearchCV(grid_pipeline, parameters, cv=5, n_jobs=-1, scoring='accuracy')

gs_clf = gs_clf.fit(docs_train, y_train) # Train on full training data

# The best estimator from grid search can now be used for predictions
best_clf = gs_clf.best_estimator_

# Test the best model on the test set - IMPORTANT STEP!
test_predictions = best_clf.predict(docs_test)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Grid Search Best CV Score: {gs_clf.best_score_:.3f}")
print(f"Grid Search Test Accuracy: {test_accuracy:.3f}")

# Test a new custom review
# test_review = ['Best movie ever']
positive_review = [
    "This film is absolutely fantastic! The acting was superb, the storyline was engaging from start to finish, and the cinematography was breathtaking. I was completely captivated throughout the entire movie and would highly recommend it to anyone who loves great cinema."
]
negative_review = [
    "What a complete waste of time. The plot made no sense, the characters were poorly developed, and the dialogue was cringe-worthy. I found myself checking my watch every five minutes waiting for this boring mess to finally end."
]

# Test the positive review
positive_pred = gs_clf.predict(positive_review)
print(f"Positive review: '{positive_review[0][:50]}...' => {movie.target_names[positive_pred[0]]}")

# Test the negative review
negative_pred = gs_clf.predict(negative_review)
print(f"Negative review: '{negative_review[0][:50]}...' => {movie.target_names[negative_pred[0]]}")

# prediction = gs_clf.predict(test_review)
# print(f"Review '{test_review[0]}' => {movie.target_names[prediction[0]]}")

# Best parameters
print("\nBest parameters from grid search:")
for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {gs_clf.best_params_[param_name]}")


"""
# test a new custom review
print(movie.target_names[gs_clf.predict(['Best movie ever'])[0]])

# best mean score
print(gs_clf.best_score_)

# the best parameters setting corresponding to that score
for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

"""


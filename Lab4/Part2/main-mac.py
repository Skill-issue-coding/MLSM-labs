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
text_clf = Pipeline([
 ('vect', CountVectorizer(min_df=2, max_features=3000, stop_words='english')),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

# train the model
text_clf.fit(docs_train, y_train)

# Evaluation of the performance on the test set
docs_test = y_test
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == y_test.target))

# SVM pipeline
text_clf = Pipeline([
 ('vect', CountVectorizer(min_df=2, max_features=3000, stop_words='english')),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])

# Evaluation of the performance on the test set
text_clf.fit(docs_train, y_train)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == y_test.target))

# more detailed performance analysis of the results
print(metrics.classification_report(movie.target, predicted,
 target_names=movie.target_names))

# confusion matrix
print(metrics.confusion_matrix(movie.target, predicted)) # [[TP, FP], [FN, TN]]

""" Grid Search """

# parameters for grid search
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2), (1,3),(2,3)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3, 1e-4),
}

# parallelize search on multiple CPU cores
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# perform the search on a smaller subset of the training data
gs_clf = gs_clf.fit(docs_train, y_train)

# test a new custom review
print(movie.target_names[gs_clf.predict(['Best movie ever'])[0]])

# best mean score
print(gs_clf.best_score_)

# the best parameters setting corresponding to that score
for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

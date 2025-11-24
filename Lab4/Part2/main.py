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

# categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
# twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
# twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

# text_clf = Pipeline([
#  ('vect', CountVectorizer()),
#  ('tfidf', TfidfTransformer()),
#  ('clf', MultinomialNB()),
# ])

# text_clf.fit(twenty_train.data, twenty_train.target)

# docs_test = twenty_test.data
# predicted = text_clf.predict(docs_test)
# print("multinomialBC accuracy ",np.mean(predicted == twenty_test.target))

# # training SVM classifier
# text_clf = Pipeline([
#  ('vect', CountVectorizer()),
#  ('tfidf', TfidfTransformer()),
#  ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
# ,max_iter=5, tol=None)),
# ])
# text_clf.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf.predict(docs_test)
# print("SVM accuracy ",np.mean(predicted == twenty_test.target))

# print(metrics.classification_report(twenty_test.target, predicted,
#  target_names=twenty_test.target_names))

# print(metrics.confusion_matrix(twenty_test.target, predicted))

# parameters = {
#  'vect__ngram_range': [(1, 1), (1, 2)],
#  'tfidf__use_idf': (True, False),
#  'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1) # see how many cores there is and use all
# s_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])
# print(gs_clf.best_score_)

# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

MOVIE_DIR: str = 'Lab4/Part2/data/movie_reviews' # could be 'data/movie_reviews'
movie = load_files(MOVIE_DIR, shuffle=True)

# Testing / Printing data
print(len(movie.data))
print(movie.target_names)
print(movie.data[0][:500])
print(movie.filenames[0])
print(movie.target[0])

# docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size=0.20, random_state=12)
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
# docs_train_counts = movieVzer.fit_transform(docs_train)

# print(movieVzer.vocabulary_.get('screen'))
# print(movieVzer.vocabulary_.get('seagal'))
# print(docs_train_counts.shape)

# movieTfmer = TfidfTransformer()
# docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# print(docs_train_tfidf.shape)

# docs_test_counts = movieVzer.transform(docs_test)
# docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# clf = MultinomialNB()
# clf.fit(docs_train_tfidf, y_train)

# y_pred = clf.predict(docs_test_tfidf)
# print(metrics.accuracy_score(y_test, y_pred))
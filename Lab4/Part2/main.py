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

MOVIE_DIR: str = 'data/movie_reviews' # could be 'data/movie_reviews'
movie = load_files(MOVIE_DIR, shuffle=True)

# Testing / Printing data
print(len(movie.data))
print(movie.target_names)
print(movie.data[0][:500])
print(movie.filenames[0])
print(movie.target[0])

docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size=0.20, random_state=12)
movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
docs_train_counts = movieVzer.fit_transform(docs_train)

print(movieVzer.vocabulary_.get('screen'))
print(movieVzer.vocabulary_.get('seagal'))
print(docs_train_counts.shape)

movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

print(docs_train_tfidf.shape)

docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

y_pred = clf.predict(docs_test_tfidf)
print(metrics.accuracy_score(y_test, y_pred))
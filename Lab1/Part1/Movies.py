import sklearn
from sklearn.datasets import load_files
import nltk
from nltk.corpus import movie_reviews

# Download the dataset (only needed once)
nltk.download('movie_reviews')

# Now you can access the movie reviews directly — no need for load_files()
movie = {
    'data': [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()],
    'target': [0 if fileid.startswith('neg') else 1 for fileid in movie_reviews.fileids()],
    'target_names': movie_reviews.categories(),
    'fileids': movie_reviews.fileids()
}

len(movie['data'])

# target names ("classes") are automatically generated from subfolder names
movie['target_names']

# First file seems to be about a Schwarzenegger movie.
movie['data'][0][:500]

# first file is in "neg" folder
movie['fileids'][0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie['target'][0]

# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Turn off pretty printing of jupyter notebook... it generates long lines
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.display_formatter.formatters['text/plain'].pprint = False


# Three tiny "documents"
docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]

# Initialize a CountVectorizer to use NLTK's tokenizer instead of its
# default one (which ignores punctuation and stopwords).
fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# Fit and transform
docs_counts = fooVzer.fit_transform(docs)
fooVzer.vocabulary_
docs_counts.shape
docs_counts.toarray()

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
fooTfmer = TfidfTransformer()

# Again, fit and transform
docs_tfidf = fooTfmer.fit_transform(docs_counts)
docs_tfidf.toarray()

# A list of new documents
newdocs = ["I have a rose and a lily.", "What a beautiful day."]

# Transform new docs
newdocs_counts = fooVzer.transform(newdocs)
newdocs_counts.toarray()

# Again, transform using tfidf
newdocs_tfidf = fooTfmer.transform(newdocs_counts)
newdocs_tfidf.toarray()

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    movie['data'], movie['target'], test_size=0.20, random_state=12
)

# Initialize CountVectorizer
movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)  # use all 25K words. Higher accuracy

# Fit and transform using training text
docs_train_counts = movieVzer.fit_transform(docs_train)
movieVzer.vocabulary_.get('screen')
movieVzer.vocabulary_.get('seagal')
docs_train_counts.shape

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)
docs_train_tfidf.shape

# Transform test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Train a Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the test set results and find accuracy
y_pred = clf.predict(docs_test_tfidf)
print("Accuracy:", sklearn.metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Very short and fake movie reviews
reviews_new = [
    'This movie was excellent', 'Absolute joy ride',
    'Steven Seagal was terrible', 'Steven Seagal shone through.',
    'This was certainly a movie', 'Two thumbs up',
    'I fell asleep halfway through', "We can't wait for the sequel!!",
    '!', '?', 'I cannot recommend this highly enough',
    'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.'
]

# Turn new reviews into TF-IDF vectors
reviews_new_counts = movieVzer.transform(reviews_new)
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)

# Make predictions
pred = clf.predict(reviews_new_tfidf)

# Print results
for review, category in zip(reviews_new, pred):
    print(f'{review!r} => {movie["target_names"][category]}')

# Mr. Seagal simply cannot win!

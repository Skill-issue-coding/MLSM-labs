""" Text feature extraction TF-IDF """

""" Going to the vector space """
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

print(vectorizer)

my_stop_words={"the","is"}
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)

print(vectorizer.stop_words)
print(vectorizer.vocabulary)

smatrix = vectorizer.transform(Z)
print(smatrix)

matrix = smatrix.todense()
print(matrix)

""" Computing the tf-idf score """

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names_out()
import pandas as pd
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])
print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
print(df)

""" Document Similarity """



''' 
What is the TF-IDF measure
    TF-IDF (Term Frequency–Inverse Document Frequency) is a numerical measure that shows how important
    a word is in a document compared to a whole collection of documents.
    
    Term Frequency (TF) measures how often a word appears in a document — the more frequent, 
    the higher the value.

    Inverse Document Frequency (IDF) reduces the weight of words that appear in many documents,
    since they carry less unique meaning.

    By combining both, TF-IDF highlights words that are frequent in one document but rare across others,
    making it useful for text analysis, search engines, and keyword extraction.

How to use TF-IDF for:
– document similarity
    
    Compute TF-IDF vectors – Represent each document as a vector of TF-IDF values, where each dimension
    corresponds to a word’s importance.

    Compare vectors – Measure similarity between documents using a metric like cosine similarity, 
    which calculates the angle between two TF-IDF vectors.

    Interpret the result – A cosine similarity close to 1 means the documents are very similar in content, 
    while a value near 0 means they’re quite different. The value can be between -1 and 1, 1 meaning they
    point in the same direction (angle between them are 0). -1 means they point to the opposite direction, 
    (angle between them are 180).


– classify text
    Convert text to TF-IDF vectors – Each document is transformed into a numerical vector where each value
    reflects the importance of a word (high for distinctive words, low for common ones).

    Train a classifier – Use these TF-IDF vectors as input features for a machine learning model, in this case
    multinomial Naive Bayes but other can work also (e.g., Logistic Regression, SVM, or a Neural Network).

    Predict classes – When a new text is given, convert it to a TF-IDF vector using the same vocabulary
    and let the trained model predict its category.

'''


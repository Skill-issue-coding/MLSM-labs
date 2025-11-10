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


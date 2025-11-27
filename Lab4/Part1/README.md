# Questions

## 1. What is the TF-IDF measure

TF-IDF (Term Frequency–Inverse Document Frequency) is a numerical measure that shows how important a word is in a document compared to a whole collection of documents.

Term Frequency (TF) measures how often a word appears in a document — the more frequent, the higher the value. It then normalizes each word. (If apple appears 2 time in doc 1 and a total of 4 times in all docs -> 2/4 = 0.5)

Inverse Document Frequency (IDF) reduces the weight of words that appear in many documents, since they carry less unique meaning. By taking log of the total number of dokuments and divide it by 1 + the number of documents the term occurs. (if we have 5 docs and apple appears in 3 -> log(5/1+3) = 0.097)

By combining both (matrix multiplication), TF-IDF highlights words that are frequent in one document but rare across the total number of documents, making it useful for text analysis, search engines, and keyword extraction.

## 2. How to use TF-IDF for

**Document similarity**

Compute TF-IDF vectors – Represent each document as a vector of TF-IDF values, where each dimension corresponds to a word’s importance.

Compare vectors – Measure similarity between documents using a metric like cosine similarity, which calculates the angle between two TF-IDF vectors. The amount of times a word appears in a dokument does not contribute to the simularity.

Interpret the result – A cosine similarity close to 1 means the documents are very similar in content, while a value near 0 means they’re quite different. The value can be between -1 and 1, 1 meaning they point in the same direction (angle between them are 0). -1 means they point to the opposite direction, (angle between them are 180). In TF-IDF the value rarely are -1

**Classify text**

Convert text to TF-IDF vectors – Each document is transformed into a numerical vector where each value reflects the importance of a word (high for distinctive words, low for common ones).

Train a classifier – Use these TF-IDF vectors as input features for a machine learning model, in this case multinomial Naive Bayes but other can work also (e.g., Logistic Regression, SVM, or a Neural Network).

Predict classes – When a new text is given, convert it to a TF-IDF vector using the same vocabulary and let the trained model predict its category.

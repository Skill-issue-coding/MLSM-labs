# Questions

## 1. Explain the TextRank algorithm and how it works

Textrank divides the text into nodes, with each node being a sentence. For word extraction only nouns and adjectives are keep. Then it compares the amount of same/shared words in each node(sentence) to each other, building a graph. By comparing the words in the nodes to each other we get the edges, which are normalized so it does not favorize long sentences. Edges that are 0 does not create a connection. Then TextRank uses a variation of the pageRank walker that counts the number of weights (connections) each node has, More weights makes the node more important => higher score. The last step is that TextRank takes the nodes with highest score and uses them (in the same original order) as the summary based of how long the summery should be. 


TextRank is a graph-based ranking algorithm primarily used for extractive text summarization and keyword extraction. It’s inspired by PageRank (used by Google for ranking web pages) and works by identifying the most “important” sentences or words in a text.

1. **Core idea**

    TextRank treats sentences or words as nodes in a graph and uses edges to represent relationships or similarity. The algorithm assumes that the more “connected” a sentence or word is to other important sentences/words, the more significant it is.

2. **TextRank for Keyword Extraction**

    1. **Preprocess Text**

        - Tokenize words.

        - Remove stopwords and punctuation.

        - Optionally, stem or lemmatize words.

    2. **Build Graph**

        - Each node represents a unique word.

        - Edges connect words that co-occur within a fixed window size (e.g., 2–5 words).

        - Edge weights can represent co-occurrence frequency.

   3. **Rank Words**

        - Iterate until scores converge (usually 20–50 iterations).

        - The top-scoring words are considered keywords.

3. **TextRank for Sentence Extraction (Summarization)**

    1. **Preprocess Text**

        - Split text into sentences.

        - Tokenize and normalize sentences (optional: remove stopwords).

    2. **Build Graph**

        - Nodes = sentences.

        - Edges = similarity between sentences (e.g., cosine similarity of sentence vectors).

        - Edge weights = degree of similarity.

    3. **Score Sentences**

        - Use the same iterative formula as above.

        - Sentences that are “referenced” by other important sentences get higher scores.

    4. **Generate Summary**

        - Pick the top _𝑁_ scoring sentences.

        - Sort them in the order they appear in the text for readability.

4. **Advantages**

    - **Unsupervised:** Doesn’t require labeled data.

    - **Language-independent:** Works for any language with tokenization.

    - **Flexible:** Can be used for both keywords and summaries.

5. **Limitations**

    - **Context Ignorance:** Doesn’t understand meaning beyond co-occurrence.

    - **Graph Construction Sensitive:** Choice of window size, similarity metric, and preprocessing affects results.

    - **Extractive Only:** Summaries are made of original sentences; it cannot paraphrase.

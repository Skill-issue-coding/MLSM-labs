# Questions

## 1. Explain the TextRank algorithm and how it works

TextRank is a graph-based ranking algorithm primarily used for extractive text summarization and keyword extraction. It’s inspired by PageRank (used by Google for ranking web pages) and works by identifying the most “important” sentences or words in a text. Let’s break it down step by step.

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

    3. **Score Nodes**

        TextRank iteratively computes a score for each node using the formula:

        ![Equation](https://latex.codecogs.com/svg.image?S(V_i)=(1-d)+d\sum_{V_j%20\in%20In(V_i)}%20\frac{w_{ji}}{\sum_{V_k%20\in%20Out(V_j)}%20w_{jk}}%20S(V_j))

        Where:  
        - ![S(V_i)](https://latex.codecogs.com/svg.image?&space;S(V_i)): score of node i

        - ![d](https://latex.codecogs.com/svg.image?d): damping factor (usually 0.85)

        - ![InV_i](https://latex.codecogs.com/svg.image?In(V_i)): nodes pointing to Vi

        - ![W_j_i](https://latex.codecogs.com/svg.image?W_j_i): weight of edge from \(V_j\) to \(V_i\)  

        - ![sum_V_k epsilon Out(V_j)](https://latex.codecogs.com/svg.image?\sum_{V_k\epsilon&space;Out(V_j)}^{}w_j_k): sum of outgoing edge weights from \(V_j\)  

    4. **Rank Words**

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

4. **Why It Works**

    - Sentences/words that share content with many others are more central.

    - Graph structure captures interconnections without relying on deep semantics.

    - Iterative scoring naturally balances importance across the entire text.

5. **Advantages**

    - **Unsupervised:** Doesn’t require labeled data.

    - **Language-independent:** Works for any language with tokenization.

    - **Flexible:** Can be used for both keywords and summaries.

6. **Limitations**

    - **Context Ignorance:** Doesn’t understand meaning beyond co-occurrence.

    - **Graph Construction Sensitive:** Choice of window size, similarity metric, and preprocessing affects results.

    - **Extractive Only:** Summaries are made of original sentences; it cannot paraphrase.

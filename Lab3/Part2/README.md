## 1. Explain what the program does

This program takes the top half of face images as input and uses machine learning to predict the bottom half. It trains four different models to complete missing facial features, then compares how well each method reconstructs the full face on people it has never seen before. The goal is to see which algorithm best understands general facial structure to fill in missing parts.

## 2. What is your interpretation of the final plot? Which algorithm has better performance

Based of our interpretation the method using extra trees is the overall most accurate, but sometimes the other methods can produce a more satisfying result.

## 3. Download the code from the link above and modify it by adding the results of the following

When looking at the result we see that regression decision tree has a clearer prediction (less noise). Extra trees is still the best predictor, but sometimes the random forest outperforms it. For example on the last person. Somehow regression decision tree can sometimes change the ethnicity of a person add facial hair.

## 4. How could performance of random forest be improved? (Hint: have a look at the [example of using Haar-like feature in face detection here](https://realpython.com/traditional-face-detection-python/))

Random Forest performance can be improved by:

    1. Better Features:
        Create smarter features (like Haar features do for faces) instead of just using raw data

    2. Feature Selection:
        Keep only the most important features to reduce noise

    3. Parameter Tuning:
        Optimize tree depth, number of trees, and split criteria

    4. Data Preparation:
        Handle imbalanced data and scale features properly

    5. Model Combinations:
        Use multiple forests together or with other algorithms

    Key Insight:
    Like Haar features show, intelligent feature design often matters more than complex models.
    Focus on creating better features rather than just making the model bigger.

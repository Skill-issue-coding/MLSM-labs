# Questions

## 1. What is the basic idea/intuition of SVM?

The basic idea behind Support Vector Machines (SVM) is to find the best separating boundary between different classes, by maximizing the margin — the distance between the decision boundary and the nearest data points from each class (called support vectors). Rather than just any separating line (in 2D) or plane (in higher dimensions), SVM chooses the one that leaves the widest possible margin between classes, which helps improve generalization on new data.

## 2. What can you do if the dataset is not linearly separable?

If the dataset is not linearly separable, SVM uses the kernel trick to project data into a higher-dimensional space where a linear separator may exist. Common kernel functions (like the radial basis function, RBF), implicitly map data to this higher space without explicitly computing the transformation, making nonlinear classification efficient.

## 3. Explain the concept of Soften Margins

When data cannot be perfectly separated — e.g., due to overlap or noise — SVM allows some points to violate the margin using a soft margin.

**This flexibility is controlled by a parameter C:**

**A large C →** hard margin (strict separation, less tolerance for errors).

**A small C →** soft margin (more tolerance for misclassified or margin-crossing points).

The value of C balances between maximizing the margin and minimizing classification errors.

## 4. What are the pros and cons of SVM?

**Pros:**

- Compact models — depend on few support vectors.

- Fast prediction once trained.

- Works well with high-dimensional data.

- Very versatile through kernel methods (can model nonlinear boundaries).

**Cons:**

- Training time scales poorly (worst-case O(N³)).

- Requires careful tuning of C and kernel parameters.

- Lacks direct probabilistic interpretation.

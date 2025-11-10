## 1. In the script of the Regression section (the one before the section RFE), we apply cross validation with 10 folds. Note that the script does not make any change to the dataset. Modify the script in oder to reshuffle the rows of the data set to randomize the cross validation folds before applying the cross validation. Run again the script but on the reshuffled data set and re-calculate the MSE and R2 scores. Do you obtain a better performance?

_To know more about reshuffling You can read the part about reshuffling in Section 3.1 ”Cross-validation: evaluating estimator performance” of scikit-learn._

We gained almost a double in performance and no one is negative. It's most likely because we got the features with bigger impact
in the beginning of the process.

## 2. What happens if you do reshuffling and RFE? do you get better results than only reshuffling?

No one is negative anymore and the performance is increased.
Slightly better performance without RFE but RFE runs faster since it removes less important features.

## 3. In the section Car Evaluation Quality, we performed the evaluation metrics for linear support vector machine, naive bayes, logistic regression and k nearest neighbours. As you can see at page 18, they do have a poor performance. Find out if there are ML algorithms that perform better on the data cars.csv data set

_You may test decision trees and random forest as well as other type of SVM._

| Number | Classifier    | Acc      | Good     | Unacc    | Vgood    |
| ------ | ------------- |----------|----------|----------|----------|
| 4      | Decision tree | 0.955729 | 0.898551 | 0.985950 | 0.938462 |
| 5      | Random forest | 0.963542 | 0.898551 | 0.990909 | 0.938462 |
| 6      | SVM rbf       | 0.854167 | 0.376812 | 0.990083 | 0.815385 |
| 7      | SVM poly      | 0.632812 | 0.449275 | 0.954545 | 0.584615 |
| 8      | SVM sigmoid   | 0.341146 | 0.000000 | 0.733884 | 0.000000 |

We see that random forest has very good performance.

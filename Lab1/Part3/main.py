from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.DESCR)

print(len(cancer.data[cancer.target==1]))
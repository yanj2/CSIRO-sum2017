import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import cross_val_score

def main():
    # NOTE: 0.421
    digits = datasets.load_digits()
    clf = SVC()
    scores = cross_val_score(clf, digits.data, digits.target)
    # NOTE: 0.973
    # iris = datasets.load_iris()
    # clf = SVC()
    # scores = cross_val_score(clf, iris.data, iris.target)
    print(scores, scores.mean())

if __name__ == "__main__":
    main()

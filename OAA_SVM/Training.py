import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

from Visualize_OAA import visualize_oaa_svm

# Create the Dataset
n = 300
X, Y = datasets.make_classification(n, n_features=2, n_redundant=0, n_classes=4,
                                    n_clusters_per_class=1, class_sep=1.3, random_state=65321)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=12345)

# List to store several SVM classifiers
svm_classifiers = []

# Train multiple binary SVM classifiers
for label in set(Y_train):
    # preprocess the labels by creating a new list that generates itself with a condition
    y_processed = [1 if y_label == label else 0 for y_label in Y_train]

    # Here I use scikit-learns svm since it is a normal linear-svm and I already implemented one.
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_processed)

    svm_classifiers.append(clf)

# Get predictions for all classifiers
y_pred = []

for x in X_test:
    # We predict each sample separately instead of all at once, this way we have an array like [0,0,1] that tells us
    # that this sample probably belongs to class 2.
    predictions = [svm.predict([x])[0] for svm in svm_classifiers]

    # numpy.argmax is simply a function that returns the index (not the value!) of the highest number in the input.
    y_pred.append(np.argmax(predictions))

weights = [svm_classifier.coef_[0] for svm_classifier in svm_classifiers]
bias = [svm_classifier.intercept_[0] for svm_classifier in svm_classifiers]

visualize_oaa_svm(X_train, Y_train, weights, bias)

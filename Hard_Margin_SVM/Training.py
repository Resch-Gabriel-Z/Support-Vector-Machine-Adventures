import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

from Hyperparameters import hyperparameters
from SVM import SVM
from Visualizer import visualize_svm

# Create Metadata
path_of_model = '-'
name_of_model = '-'
save_model = False
test_model = True

X, y = datasets.make_blobs(
    n_samples=300, n_features=2, centers=2, cluster_std=1.05, random_state=254
)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

SVM_hyperparameters = [hyperparameters['learning_rate'], hyperparameters['training_steps'],
                       hyperparameters['batch_size']]

clf = SVM(*SVM_hyperparameters, optim='GD')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Save the trained model in datafile
if save_model:
    model = {'weights': clf.weights,
             'bias': clf.bias}

    df = pd.DataFrame(model)
    df.to_csv(f'{path_of_model}/{name_of_model}', index=False)

if test_model:
    visualize_svm(X, y, clf.weights, clf.bias)

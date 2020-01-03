from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml
import numpy as np


def _subsample_mnist(images, labels, n):

    indices_to_keep = np.array([], dtype=np.int)
    classes = np.unique(labels)
    n_classes = len(classes)
    keep_by_class = int(n / n_classes)
    for i in range(10):
        if i in classes:
            belongs_to_class_i = np.where(labels==i)[0]
            indices_to_keep = np.append(indices_to_keep, belongs_to_class_i[:keep_by_class])

    X = images[indices_to_keep]
    Y = labels[indices_to_keep]
    return X, Y


def load_mnist():
    """Load the MNIST dataset."""

    mnist = fetch_openml('mnist_784')
    images =  mnist['data'][:10000]
    labels = np.array(list(map(int, mnist['target'][:10000])))

    classes = np.unique(labels)
    n_clusters_mnist = len(classes)
    keep_per_class = 300
    n = keep_per_class * len(classes)
    images = np.array(images)
    labels = np.array(labels)

    return _subsample_mnist(images, labels, n)


def logistic_grid_search(train_features, train_labels):
    # Create first pipeline for base without reducing features.

    pipe = Pipeline([('classifier' , RandomForestClassifier())])
    # pipe = Pipeline([('classifier', RandomForestClassifier())])

    # Create param grid.

    param_grid = [
        {'classifier' : [LogisticRegression()],
         'classifier__penalty' : ['l1', 'l2'],
        'classifier__C' : np.logspace(-4, 4, 20),
        'classifier__solver' : ['liblinear']},
    #     {'classifier' : [RandomForestClassifier()],
    #     'classifier__n_estimators' : list(range(10,101,10)),
    #     'classifier__max_features' : list(range(6,32,5))}
    ]

    # Create grid search object

    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

    # Fit on data

    best_clf = clf.fit(train_features, train_labels.numpy())
    
    return best_clf
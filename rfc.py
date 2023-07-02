import numpy as np
from helper_functions import add_status_bar
from multiprocessing import Pool

from dtc import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the random forest.
    max_depth : int, default=100
        The maximum depth of each decision tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    criterion : str, default='entropy'
        The function to measure the quality of a split.

    Attributes
    ----------
    n_estimators : int
        The number of decision trees in the random forest.
    max_depth : int
        The maximum depth of each decision tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    criterion : str
        The function to measure the quality of a split.
    estimators : list
        The list of trained decision tree classifiers.

    """

    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, criterion="entropy"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.estimators = []

    def _train_estimator(self, X, y):
        """
        Trains a decision tree classifier on the given data.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target labels.
        attributes : array-like, optional
            Subset of attributes to consider for splitting. Default is None.

        Returns
        -------
        DecisionTreeClassifier
            The trained decision tree classifier.

        """
        estimator = DecisionTreeClassifier(max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           criterion=self.criterion)
        estimator.fit(X, y)
        return estimator

    def fit(self, X, y, replace=True):
        """
        Fits the random forest classifier to the training data.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target labels.
        replace : bool, default=True
            Whether to sample with replacement.

        """
        with Pool() as pool:
            results = []
            for _ in add_status_bar(range(self.n_estimators)):
                indices = np.random.choice(range(len(X)), size=len(X), replace=replace)
                X_bootstrapped = X[indices]
                y_bootstrapped = y[indices]
                results.append(pool.apply_async(self._train_estimator, (X_bootstrapped, y_bootstrapped)))
            self.estimators = [result.get() for result in add_status_bar(results)]

    def predict(self, X, get_pred_probs=False):
        """
        Predicts the target labels for the input features.

        Parameters
        ----------
        X : array-like
            The input features.
        get_pred_probs : bool, default=False
            Whether to return prediction probabilities. If True, returns the average of prediction probabilities.

        Returns
        -------
        array-like
            The predicted target labels.

        """
        trees_predictions = np.array([[tree.predict(tree.root, x) for x in X] for tree in self.estimators])
        if get_pred_probs:
            return trees_predictions.mean(axis=0)
        return np.round(trees_predictions.mean(axis=0)).astype(int)



if __name__ == "__main__":
    # Load the training and testing data
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(np.int32)
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int32)

    # Split data into features and labels
    X_train, y_train = training_spam[:, :-1], training_spam[:, -1]
    X_test, y_test = testing_spam[:, :-1], testing_spam[:, -1]

    # Initialize the model
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)

    # Fit the model
    rfc.fit(X_train, y_train)

    # Make predictions
    y_pred = rfc.predict(X_test)

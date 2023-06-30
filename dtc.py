import numpy as np
from collections import Counter
from pprint import pprint
from helper_functions import visualize_tree



# Define the class DecisionTreeClassifier
class DecisionTreeClassifier:
    """
    A decision tree classifier.

    This class implements a simple decision tree binary classifier using the ID3
    algorithm. It supports training and prediction.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_depth : int, default=100
        The maximum depth of the tree.
    n_feature : int, optional
        The number of features to consider when looking for the best split.
        If None, then `n_feature=n_features`.
    criterion : str, default='entropy'
        The criterion to use for computing attribute importance.
        Options are 'entropy' for information gain and 'gini' for Gini index.

    Attributes
    ----------
    root : dict
        The root of the decision tree.
    default_class : int or None
        The default class label, defined as the most common class label in the dataset.

    """
    def __init__(self, min_samples_split=2, max_depth=100, max_feature=None, criterion='entropy'):
        # Initialize the object with minimum sample split, maximum depth, number of features, and criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.root = None  # Initialize root of the tree as None
        self.default_class = None  # Initialize default class as None
        self.criterion = criterion.lower()  # Convert criterion to lowercase

    def fit(self, X, y, attributes=None):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.

        """
        # Set the default class as the most common class in y
        self.default_class = self._plurality_value(y)
        # Build the decision tree
        self.root = self.build_tree(X, y, attributes)

    def predict(self, tree, x):
        """
        Predict class for x.

        The predicted class of an input sample is computed as per the decision
        tree.

        Parameters
        ----------
        tree : dict
            The decision tree to use for prediction.
        x : array-like of shape (n_features,)
            The input sample.

        Returns
        -------
        int or str
            The predicted class.
        """
        for attribute, branches in tree.items():
            x_value = x[attribute]
            if x_value in branches:
                subtree = branches[x_value]
                if isinstance(subtree, dict):  # If the subtree is a dict, recurse
                    return self.predict(subtree, x)
                else:  # If the subtree is a leaf, return the class label
                    return subtree

        # Return the default class if no prediction can be made
        return self.default_class

    def build_tree(self, X, y, attributes=None, depth=0):
        """
        Recursive function to build the decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        attributes : array-like of shape (n_features,), optional
            The attribute indices to consider. If None, all attributes are considered.
        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        dict
            The decision tree.
        """
        if attributes is None:
            attributes = np.arange(X.shape[1])


        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Base case: stop if the tree is too deep,
        # there are too few samples, there is only one class left or no feature left
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1 or n_features == 0:
            return self._plurality_value(y)

        # Compute the importance of each attribute
        if self.criterion == 'entropy':
            attributes_importance = [self._importance(i, X, y) for i in range(n_features) if i in attributes]
        elif self.criterion == 'gini':
            attributes_importance = [self._gini_importance(i, X, y) for i in range(n_features) if i in attributes]
        else:
            raise ValueError("Invalid criterion. Supported criteria are 'entropy' and 'gini'.")

        # If no attributes are left, return a leaf node
        if not attributes_importance:
            return self._plurality_value(y)

        # Choose the attribute with the highest importance
        best_attribute_idx = np.argmax(attributes_importance)

        best_attribute = attributes[best_attribute_idx]
        best_attribute_values = np.unique(X[:, best_attribute_idx])

        # Create a new internal node for this attribute
        tree = {best_attribute: {}}
        for value in best_attribute_values:
            indices = X[:, best_attribute_idx] == value
            X_new = np.delete(X[indices], best_attribute_idx, axis=1)
            y_new = y[indices]
            new_attributes = np.delete(attributes, best_attribute_idx)
            # Recursively build the subtrees
            subtree = self.build_tree(X_new, y_new, new_attributes, depth + 1)
            tree[best_attribute][value] = subtree

        return tree

    def _plurality_value(self, labels):
        """
        Compute the most common class label.

        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            The class labels.

        Returns
        -------
        int or str
            The most common class label.
        """
        return Counter(labels).most_common(1)[0][0]

    def _importance(self, attribute_index, X, y):
        """
        Compute the importance of an attribute using information gain.

        Parameters
        ----------
        attribute_index : int
            The index of the attribute.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        float
            The information gain of the attribute.
        """
        p = np.count_nonzero(y == 1)
        n = np.count_nonzero(y == 0)
        info_gain = self._B(p, n) - self._remainder(attribute_index, X, y)
        return info_gain

    def _gini_importance(self, attribute_index, X, y):
        """
        Compute the importance of an attribute using Gini index.

        Parameters
        ----------
        attribute_index : int
            The index of the attribute.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        float
            The Gini index of the attribute.
        """
        attribute_values = X[:, attribute_index]
        p = np.count_nonzero(y == 1)
        n = np.count_nonzero(y == 0)
        total = n + p

        gini_index = 0.0
        for value in np.unique(attribute_values):
            value_labels = y[attribute_values == value]
            p_k = np.count_nonzero(value_labels == 1)
            n_k = np.count_nonzero(value_labels == 0)
            proportion = (p_k + n_k) / total
            gini_index += proportion * (1 - (p_k / (p_k + n_k))**2 - (n_k / (p_k + n_k))**2)

        return gini_index

    def _B(self, p, n):
        """
        Compute the entropy of a binary distribution.

        Parameters
        ----------
        p : int
            The number of positive samples.
        n : int
            The number of negative samples.

        Returns
        -------
        float
            The entropy of the binary distribution.
        """
        epsilon = 1e-10
        q = p / (p + n)
        return -(q * np.log2(q + epsilon) + (1 - q) * np.log2(1 - q + epsilon))

    def _remainder(self, attribute_index, X, y):
        """
        Compute the expected entropy after splitting on the given attribute.

        Parameters
        ----------
        attribute_index : int
            The index of the attribute.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        float
            The expected entropy.
        """
        attribute_values = X[:, attribute_index]
        p = np.count_nonzero(y == 1)
        n = np.count_nonzero(y == 0)
        total = n + p

        remainder = 0.0
        for value in np.unique(attribute_values):
            value_labels = y[attribute_values == value]
            p_k = np.count_nonzero(value_labels == 1)
            n_k = np.count_nonzero(value_labels == 0)
            proportion = (p_k + n_k) / total
            entropy = self._B(p_k, n_k)
            remainder += proportion * entropy

        return remainder

    def prune(self, X, y, tree, randomize=True, random_seed=None):
        """
        Prune the decision tree using the validation set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The validation input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        tree : dict
            The decision tree to be pruned.
        randomize : bool, default=True
            Whether to introduce randomness during pruning.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            The pruned decision tree.
        """
        if randomize:
            np.random.seed(random_seed)

        if not isinstance(tree, dict):
            # If the subtree is a leaf node, return it
            return tree

        for attribute, branches in tree.items():
            for value, subtree in branches.items():
                # Prune the subtree recursively
                tree[attribute][value] = self.prune(X, y, subtree, randomize, random_seed)

        if self._can_prune(tree, X, y) and (not randomize or np.random.random() < 0.5):
            # If the tree can be pruned and the randomness condition is satisfied, replace the subtree with the majority class
            class_labels = [self.predict(tree, x) for x in X]
            majority_class = Counter(class_labels).most_common(1)[0][0]
            return majority_class

        return tree

    def _can_prune(self, tree, X, y):
        """
        Check if the decision tree can be pruned.

        The decision tree can be pruned if the error rate on the validation set
        is not significantly worse than the error rate without pruning.

        Parameters
        ----------
        tree : dict
            The decision tree.
        X : array-like of shape (n_samples, n_features)
            The validation input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        bool
            True if the tree can be pruned, False otherwise.
        """
        # Calculate the error rate without pruning
        predictions = [self.predict(tree, x) for x in X]
        error_rate_without_pruning = sum(y != predictions) / len(y)

        # Calculate the error rate with pruning
        pruned_tree = self.prune_tree(tree, X, y)
        pruned_predictions = [self.predict(pruned_tree, x) for x in X]
        error_rate_with_pruning = sum(y != pruned_predictions) / len(y)

        # Check if the error rate with pruning is not significantly worse
        return error_rate_with_pruning >= error_rate_without_pruning

    def prune_tree(self, tree, X, y):
        """
        Prune the decision tree recursively.

        This method creates a pruned copy of the decision tree without modifying
        the original tree.

        Parameters
        ----------
        tree : dict
            The decision tree.
        X : array-like of shape (n_samples, n_features)
            The validation input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        dict
            The pruned decision tree.
        """
        pruned_tree = {}
        for attribute, branches in tree.items():
            pruned_tree[attribute] = {}
            for value, subtree in branches.items():
                if isinstance(subtree, dict):
                    pruned_subtree = self.prune_tree(subtree, X, y)
                    if not isinstance(pruned_subtree, dict):
                        # If the subtree is pruned to a leaf node, replace the subtree with the majority class
                        class_labels = [self.predict(tree, x) for x in X]
                        majority_class = Counter(class_labels).most_common(1)[0][0]
                        pruned_tree[attribute][value] = majority_class
                    else:
                        pruned_tree[attribute][value] = pruned_subtree
                else:
                    pruned_tree[attribute][value] = subtree

        return pruned_tree


if __name__ == "__main__":
    # Load the spam dataset
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(np.int32)
    validation_spam = training_spam[-100:]
    training_spam = training_spam[:-100]
    # Create and train a decision tree
    decision_tree = DecisionTreeClassifier(max_depth=10, criterion='gini')
    decision_tree.fit(training_spam[:, 1:], training_spam[:, 0])
    # Prune the tree
    # decision_tree.prune(validation_spam[:, 1:], validation_spam[:, 0], decision_tree.root)
    # Visualize the decision tree
    dot = visualize_tree(decision_tree.root)
    dot.render('tree_visualization', view=True)
    # Load the testing data
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int32)
    # Predict the labels of the testing data
    predictions = [decision_tree.predict(decision_tree.root, instance) for instance in testing_spam[:, 1:]]
    print(predictions)
    # Print the decision tree
    pprint(decision_tree.root)

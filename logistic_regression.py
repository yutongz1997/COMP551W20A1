import numpy as np
import tools


class LogisticRegression:
    """
    A logistic regression based binary classifier with L2 regularization. All categorical features should
    be one-hot encoded.

    Usage:
    ------
    For example, we would like to learn a binary classification dataset, where column 0 (3 categories) and
    column 2 (2 categories) are discrete features, and column 1 is a continuous feature (as below)::

        X_train = np.array([[1, 2.315, 0],
                            [0, 1.744, 0],
                            [2, 0.289, 1],
                            [2, 4.413, 0]])
        y_train = np.array([0, 1, 1, 0])

    The design matrix X_train should be transformed using one-hot encoding into::

        X_train_new = np.array([[0, 1, 0, 2.315, 0],
                                [1, 0, 0, 1.744, 0],
                                [0, 0, 1, 0.289, 1],
                                [0, 0, 1, 4.413, 0]])

    (if values of continuous features are large, it is best to normalize them.) The following code could
    fit a model and make predictions::

        from logistic_regression import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train_new, y_train)
        y_pred = lr.predict(X_test_new)
    """

    def __init__(self, learning_rate=1e-3, regularization_param=1.0, max_iterations=10000, change_threshold=1e-9):
        """
        Initialize all attributes of this logistic regression binary classifier

        :param learning_rate: the learning rate of gradient descent (= 1e-3 by default)
        :param regularization_param: the L2 regularization parameter (= 1.0 by default)
        :param max_iterations: the maximum iterations of gradient descent (= 10000 by default)
        :param change_threshold: the stopping threshold for change in cost function
        """

        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iterations = max_iterations
        self.change_threshold = change_threshold
        # The weight vector
        self.weight = np.zeros(1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit a logistic regression model given training data

        :param X_train: the training design matrix
        :param y_train: the array of training labels
        """

        num_samples, num_features = X_train.shape
        self.weight = np.zeros(shape=num_features, dtype=float)
        iteration = 1
        # Record the cost function change
        cost_prev = 0.0
        cost_change = np.inf
        # Run gradient descent iterations
        while iteration <= self.max_iterations and cost_change > self.change_threshold:
            logit = np.dot(X_train, self.weight)
            y_pred = tools.logistic(logit)
            # Compute the current gradient with L2 regularization term
            gradient = (np.dot(X_train.T, y_pred - y_train) + self.regularization_param * self.weight) / num_samples
            # Update weights in the direction of gradient
            self.weight -= self.learning_rate * gradient
            # Increment the iteration count
            iteration += 1
            # Compute the current cost using cross-entropy loss
            cost_curr = (-np.sum(y_train * np.log1p(np.exp(-logit)) + (1 - y_train) * np.log1p(np.exp(logit)))
                         + 0.5 * self.regularization_param * np.linalg.norm(self.weight)) / num_samples
            cost_change = np.abs(cost_curr - cost_prev)
            cost_prev = cost_curr

    def predict(self, X_test: np.ndarray):
        """
        Predict labels for unseen data using the trained model

        :param X_test: the test design matrix
        :return: an array of predicted labels
        """

        y_pred = tools.logistic(np.dot(X_test, self.weight))
        for i in range(y_pred.shape[0]):
            y_pred[i] = 1 if y_pred[i] >= 0.5 else 0
        return y_pred.astype(int)

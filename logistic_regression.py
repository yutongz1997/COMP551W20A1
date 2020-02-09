import numpy as np
import tools


class LogisticRegression:
    """
    Logistic regression
    """

    def __init__(self, learning_rate=1e-3, regularization_param=1.0, max_iterations=10000, threshold=1e-9):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.weight = np.zeros(1)

    def fit(self, X, y):
        # Ensure there is no dimension mismatch
        num_samples, num_features = X.shape
        assert y.shape[0] == num_samples
        self.weight = np.zeros(shape=num_features, dtype=float)

        iteration = 1
        # Record the cost function change
        cost_prev = 0.0
        cost_change = np.inf

        # Run gradient descent iterations
        while iteration <= self.max_iterations and cost_change > self.threshold:
            # print(f'Current iteration: {iteration}')
            logit = np.dot(X, self.weight)
            y_pred = tools.logistic(logit)
            gradient = (np.dot(X.T, y_pred - y) + self.regularization_param * self.weight) / num_samples
            self.weight = self.weight - self.learning_rate * gradient
            # Increment the iteration count
            iteration += 1
            # Calculate the change in cost function
            cost_curr = (-np.sum(y * np.log1p(np.exp(-logit)) + (1 - y) * np.log1p(np.exp(logit)))
                + 0.5 * self.regularization_param * np.linalg.norm(self.weight)) / num_samples
            cost_change = np.abs(cost_curr - cost_prev)
            cost_prev = cost_curr
        print(f'Number of training iterations: {iteration - 1}, cost change: {cost_change}')

    def predict(self, X):
        assert X.shape[1] == self.weight.shape[0]
        y_pred = tools.logistic(np.dot(X, self.weight))
        for i in range(y_pred.shape[0]):
            y_pred[i] = 1 if y_pred[i] >= 0.5 else 0
        return y_pred.astype(int)

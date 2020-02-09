import numpy as np
import tools


def k_fold(X: np.ndarray, y: np.ndarray, model, k=5):
    """
    Run k-fold cross validation to check the average model accuracy

    :param X: the design matrix
    :param y: the array of labels
    :param model: the model (which must contain fit() and predict() methods) to be cross validated
    :param k: the number of folds (= 5 by default)
    :return: the average model accuracy after k-fold cross validation
    """

    num_samples = X.shape[0]
    k_accuracies = np.zeros(shape=k)
    for i in range(k):
        # Obtain the indices of training and validation data
        validation_range = list(range(int(i / k * num_samples), int((i + 1) / k * num_samples)))
        training_range = list(set(range(num_samples)) - set(validation_range))
        # Split the data into training and validation sets
        X_train = X[training_range, :]
        y_train = y[training_range]
        X_val = X[validation_range, :]
        y_val = y[validation_range]
        # Fit the model and make predictions using the current training-validation split setting
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # Evaluate and store the current model accuracy
        k_accuracies[i] = tools.evaluate_acc(y_val, y_pred)
    avg_accuracy = k_accuracies.mean()
    return avg_accuracy, k_accuracies

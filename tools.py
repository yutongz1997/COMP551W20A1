import numpy as np


def logistic(z):
    """
    Compute the value of logistic function sigma(z) = 1 / (1 + exp(-z))

    :param z: the number or NumPy array
    :return: the value of logistic function sigma(z)
    """
    return 1 / (1 + np.exp(-z))


def gaussian_pdf(x, mean, var):
    """
    Compute the value of Gaussian PDF p(x|mu, sigma^2) = 1 / sqrt(2pi * sigma^2) * exp(-(x - mu)^2 / (2 * sigma^2))

    :param x: the data point
    :param mean: mean
    :param var: variance
    :return: the value of Gaussian PDF p(x|mu, sigma^2)
    """
    return 1.0 / np.sqrt(2.0 * np.pi * var) * np.exp(-0.5 * ((x - mean) ** 2) / var)


def split_dataset(dataset: np.ndarray, split_ratio=0.8):
    """
    Randomly split the dataset into training / test sets given a split ratio

    :param dataset: the dataset in which its last column are labels
    :param split_ratio: number in [0..1] representing the size ratio of training set (=0.8 by default)
    :return: the training and test sets
    """
    np.random.shuffle(dataset)
    num_training_samples = int(dataset.shape[0] * split_ratio)
    X = dataset[:, :-1].astype(float)
    y = dataset[:, -1].astype(int)
    X_train = X[:num_training_samples, :]
    X_test = X[num_training_samples:, :]
    y_train = y[:num_training_samples]
    y_test = y[num_training_samples:]
    return X_train, X_test, y_train, y_test


def evaluate_acc(y: np.ndarray, y_pred: np.ndarray):
    """
    Evaluate the model accuracy by comparing the true labels and predicted labels

    :param y: the array of true labels
    :param y_pred: the array of predicted labels
    :return: the model accuracy in [0..1] (= number of correct predictions / number of labels)
    """
    assert isinstance(y, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y.shape[0] == y_pred.shape[0]

    return (y == y_pred).sum() / y.shape[0]

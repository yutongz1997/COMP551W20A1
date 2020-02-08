import numpy as np
import tools


def cross_validate(X: np.ndarray, y: np.ndarray, model, k=5):
    """
    k-fold cross validation

    :param X:
    :param y:
    :param model:
    :param k:
    :return:
    """
    assert X.shape[0] == y.shape[0]

    num_samples = X.shape[0]
    accuracies = np.zeros(shape=k)
    for i in range(k):
        #
        validation_range = list(range(int(i / k * num_samples), int((i + 1) / k * num_samples)))
        training_range = list(set(range(num_samples)) - set(validation_range))
        #
        X_train = X[training_range, :]
        y_train = y[training_range]
        X_val = X[validation_range, :]
        y_val = y[validation_range]
        #
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        #
        accuracies[i] = tools.evaluate_acc(y_val, y_pred)
    avg_accuracy = accuracies.mean() * 100

    print(f'K-fold cross validation accuracies: {accuracies}')
    print(f'The final average accuracy is: {avg_accuracy:.1f}%')

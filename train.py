import pandas as pd
import tools
from logistic_regression import LogisticRegression
from sklearn.linear_model import SGDClassifier
from naive_bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from cross_validation import cross_validate


def load_dataset(dataset_filename: str):
    df = pd.read_csv(dataset_filename, header=None)
    df.drop(columns=[1], axis=1, inplace=True)
    dataset = df.values

    for i in range(dataset.shape[0]):
        dataset[i, -1] = 1 if dataset[i, -1] == 'g' else 0

    return tools.split_dataset(dataset, split_ratio=0.8)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset('dataset/ionosphere/ionosphere.data')

    LR = LogisticRegression(max_iterations=100000)
    # LR.fit(X_train, y_train)
    # y_pred_LR = LR.predict(X_test)
    #
    LR_baseline = SGDClassifier(max_iter=100000)
    # LR_baseline.fit(X_train, y_train)
    # y_pred_LR_baseline = LR_baseline.predict(X_test)

    # NB = NaiveBayes(X_train.shape[1])
    # NB.fit(X_train, y_train)
    # y_pred_NB = NB.predict(X_test)
    #
    # NB_baseline = GaussianNB()
    # NB_baseline.fit(X_train, y_train)
    # y_pred_NB_baseline = NB.predict(X_test)

    cross_validate(X_train, y_train, LR)
    cross_validate(X_train, y_train, LR_baseline)

    # accuracy_LR = tools.evaluate_acc(y_test, y_pred_LR) * 100
    # accuracy_LR_baseline = tools.evaluate_acc(y_test, y_pred_LR_baseline) * 100
    # accuracy_NB = tools.evaluate_acc(y_test, y_pred_NB) * 100
    # accuracy_NB_baseline = tools.evaluate_acc(y_test, y_pred_NB_baseline) * 100
    # print(f'Logistic Regression accuracy: {accuracy_LR:.1f}%, while scikit-learn gives {accuracy_LR_baseline:.1f}%')
    # print(f'Naive Bayes accuracy: {accuracy_NB:.1f}%, while scikit-learn gives {accuracy_NB_baseline:.1f}%')

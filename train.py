import numpy as np
import pandas as pd
import tools
from sklearn.preprocessing import MinMaxScaler
from logistic_regression import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from naive_bayes import NaiveBayes
from cross_validation import cross_validate


def load_dataset(dataset_filename: str):
    df = pd.read_csv(dataset_filename, header=None)
    # df.drop(columns=[1], axis=1, inplace=True)
    dataset = df.values

    # le = LabelEncoder()
    # dataset[:, -1] = le.fit_transform(dataset[:, -1])
    dataset = dataset.astype(float)

    for j in [0, 2, 4, 10, 11, 12]:
        min = np.min(dataset[:, j])
        diff = np.max(dataset[:, j]) - min
        dataset[:, j] = (dataset[:, j] - min) / diff

    # scaler = MinMaxScaler()
    # dataset = scaler.fit_transform(dataset)

    return tools.split_dataset(dataset, split_ratio=0.8)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset('dataset/adult/adult_processed.csv')

    # LR = LogisticRegression(max_iterations=100000)
    # LR.fit(X_train, y_train)
    # y_pred_LR = LR.predict(X_test)
    #
    # LR_baseline = SGDClassifier(max_iter=100000)
    # LR_baseline.fit(X_train, y_train)
    # y_pred_LR_baseline = LR_baseline.predict(X_test)

    NB = NaiveBayes(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
    NB.fit(X_train, y_train)
    y_pred_NB = NB.predict(X_test)
    #
    # NB_baseline = GaussianNB()
    # NB_baseline.fit(X_train, y_train)
    # y_pred_NB_baseline = NB.predict(X_test)

    # cross_validate(X_train, y_train, LR)
    cross_validate(X_train, y_train, NB)

    # accuracy_LR = tools.evaluate_acc(y_test, y_pred_LR) * 100
    # accuracy_LR_baseline = tools.evaluate_acc(y_test, y_pred_LR_baseline) * 100
    # accuracy_NB = tools.evaluate_acc(y_test, y_pred_NB) * 100
    # accuracy_NB_baseline = tools.evaluate_acc(y_test, y_pred_NB_baseline) * 100
    # print(f'Logistic Regression accuracy: {accuracy_LR:.1f}%, while scikit-learn gives {accuracy_LR_baseline:.1f}%')
    # print(f'Naive Bayes accuracy: {accuracy_NB:.1f}%, while scikit-learn gives {accuracy_NB_baseline:.1f}%')
    # print(f'Accuracy: {accuracy_NB:.1f}%')

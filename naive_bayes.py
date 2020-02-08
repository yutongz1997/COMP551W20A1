import numpy as np
import tools


class NaiveBayes:
    """
    A mixed Naive Bayes classifier that supports both continuous and categorical features
    """

    def __init__(self, num_features: int, categorical_features=None, num_classes=2, alpha_cont=1e-9, alpha_cat=1.0):
        self.categorical_features = list(set(categorical_features)) if categorical_features is not None else []
        self.continuous_features = list(set(range(num_features)) - set(self.categorical_features))
        # The number of classes, continuous and categorical features
        self.num_classes = num_classes
        self.num_cont_features = len(self.continuous_features)
        self.num_cat_features = len(self.categorical_features)
        # The prior probabilities of every class
        self.priors = np.zeros(self.num_classes)
        # The mean and variance matrices for Gaussian Naive Bayes
        self.gaussian_mean = np.zeros(shape=(self.num_classes, self.num_cont_features))
        self.gaussian_var = np.zeros(shape=(self.num_classes, self.num_cont_features))
        # The frequency tables for Multinomial Naive Bayes
        self.frequency_tables = []
        for i in range(self.num_cat_features):
            self.frequency_tables.append(np.zeros(1))
        # SMOOTHING CONSTANT
        self.alpha_cont = alpha_cont
        self.alpha_cat = alpha_cat

    def learn_priors(self, y_train: np.ndarray):
        """
        Compute prior probabilities for each class

        :param y_train: array of labels
        """
        self.priors = np.bincount(y_train) / y_train.shape[0]

    def extract_features_by_class(self, X: np.ndarray, y: np.ndarray, feature_type='continuous'):
        """

        :param X:
        :param y:
        :param feature_type:
        :return:
        """
        features = self.continuous_features if feature_type == 'continuous' else self.categorical_features
        #
        class_count = np.bincount(y)
        X_classes = []
        for i in range(self.num_classes):
            X_classes.append(np.zeros(shape=(class_count[i], len(features))))
        #
        row_counter = np.zeros(shape=self.num_classes, dtype=int)
        for i in range(X.shape[0]):
            cur_class = y[i]
            X_classes[cur_class][row_counter[cur_class]] = X[i, features]
            row_counter[cur_class] += 1
        return X_classes

    def compute_mean_var(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Compute mean and variance for all continuous data

        :param X_train: the design matrix
        :param y_train: the array of labels
        """
        X_classes = self.extract_features_by_class(X_train, y_train, feature_type='continuous')
        # Finally, calculate mean and standard deviations
        for i in range(self.num_classes):
            for j in range(self.num_cont_features):
                self.gaussian_mean[i, j] = np.mean(X_classes[i][:, j])
                self.gaussian_var[i, j] = np.var(X_classes[i][:, j]) + self.alpha_cont

    def compute_continuous_likelihoods(self, x: np.ndarray):
        """

        :param x:
        """
        cont_likelihoods = np.ones(shape=self.num_classes)
        for i in range(self.num_classes):
            likelihood = 1.0
            for j in range(self.num_cont_features):
                cur_value = x[self.continuous_features[j]]
                likelihood = likelihood * tools.gaussian_pdf(cur_value,
                                                             self.gaussian_mean[i, j], self.gaussian_var[i, j])
            cont_likelihoods[i] = likelihood
        return cont_likelihoods

    def build_frequency_tables(self, X_train: np.ndarray, y_train: np.ndarray):
        """

        :param X_train:
        :param y_train:
        """
        X_train_classes = self.extract_features_by_class(X_train, y_train, feature_type='categorical')
        for i in range(self.num_cat_features):
            num_categories = np.bincount(X_train[:, self.categorical_features[i]].astype(int)).shape[0]
            self.frequency_tables[i] = np.zeros(shape=(num_categories, self.num_classes))
            for j in range(self.num_classes):
                #
                category_count = np.zeros(shape=num_categories)
                for row_value in X_train_classes[j][:, i].astype(int):
                    category_count[row_value] += 1
                #
                self.frequency_tables[i][:, j] =\
                    (category_count + self.alpha_cat) / (category_count.sum() + self.alpha_cat * self.num_cat_features)

    def compute_categorical_likelihoods(self, x: np.ndarray):
        cat_likelihoods = np.ones(shape=self.num_classes)
        for i in range(self.num_classes):
            likelihood = 1.0
            for j in range(self.num_cat_features):
                cur_category = int(x[self.categorical_features[j]])
                likelihood = likelihood * self.frequency_tables[j][cur_category, i]
            cat_likelihoods[i] = likelihood
        return cat_likelihoods

    def learn_posterior(self, likelihoods: np.ndarray):
        """
        Compute (pseudo) posterior probabilities for every class.
        Notice that we do not account for the evidence term, because it has no effect on final results
        and can also save computation time

        :param likelihoods:
        :return:
        """
        posteriors = np.zeros(shape=self.num_classes)
        for i in range(self.num_classes):
            posteriors[i] = likelihoods[i] * self.priors[i]
        return posteriors

    def fit(self, X_train, y_train):
        self.learn_priors(y_train)
        self.compute_mean_var(X_train, y_train)
        self.build_frequency_tables(X_train, y_train)

    def predict(self, X_test):
        num_samples = X_test.shape[0]
        y_pred = np.zeros(shape=num_samples, dtype=int)
        for i in range(num_samples):
            cont_likelihoods = self.compute_continuous_likelihoods(X_test[i])
            cat_likelihoods = self.compute_categorical_likelihoods(X_test[i])
            posteriors = self.learn_posterior(cont_likelihoods * cat_likelihoods)
            y_pred[i] = posteriors.argmax()
        return y_pred

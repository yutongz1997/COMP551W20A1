import numpy as np
import tools


class NaiveBayes:
    """
    A mixed Naive Bayes classifier that supports both continuous and discrete features, which are assumed to
    have Gaussian and categorical distributions, respectively.

    Usage:
    ------
    For example, we would like to learn a binary classification dataset, where column 0 (3 categories) and
    column 2 (2 categories) are discrete features, and column 1 is a continuous feature (as below)::

        X_train = np.array([[1, 2.315, 0],
                            [0, 1.744, 0],
                            [2, 0.289, 1],
                            [2, 4.413, 0]])
        y_train = np.array([0, 1, 1, 0])

    (if values of continuous features are large, it is best to normalize them.) The following code could fit
    a model and make predictions::

        from naive_bayes import NaiveBayes
        mnb = NaiveBayes(categorical_features=[0, 2])
        mnb.fit(X_train, y_train)
        y_pred = mnb.predict(X_test)
    """

    def __init__(self, categorical_features=None, alpha_cont=1e-9, alpha_cat=1.0):
        """
        Initialize all attributes of this mixed Naive Bayes classifier

        :param categorical_features: an index list of all categorical features
        :param alpha_cont: the smoothing constant for continuous features (= 1e-9 by default)
        :param alpha_cat: the smoothing constant for categorical features (= 1 by default, also
        called Laplace smoothing)
        """

        # The index lists of continuous and categorical features
        self.categorical_features = list(set(categorical_features)) if categorical_features is not None else []
        self.continuous_features = []
        # The number of classes, continuous and categorical features
        self.num_classes = 0
        self.num_cont_features = 0
        self.num_cat_features = len(self.categorical_features)
        # The prior probabilities of every class
        self.priors = np.zeros(1)
        # The mean and variance matrices for Gaussian Naive Bayes
        self.gaussian_mean = np.zeros(1)
        self.gaussian_var = np.zeros(1)
        # The frequency tables for categorical Naive Bayes
        self.frequency_tables = []
        for i in range(self.num_cat_features):
            self.frequency_tables.append(np.zeros(1))
        # The smoothing constants to avoid division by zero in probability computations
        self.alpha_cont = alpha_cont
        self.alpha_cat = alpha_cat
        # The training data separated by class and by continuous / categorical features
        self.X_train_classes_cont = []
        self.X_train_classes_cat = []

    def learn_priors(self, y_train: np.ndarray):
        """
        Compute class prior probabilities for each class

        :param y_train: the array of training labels
        """

        self.priors = np.bincount(y_train) / y_train.shape[0]

    def extract_features_by_class(self, X_train: np.ndarray, y_train: np.ndarray, feature_type='continuous'):
        """
        Extract and separate training data by class and by specified feature type

        :param X_train: the training design matrix
        :param y_train: the array of training labels
        :param feature_type: the training data with this feature type will be extracted (either 'continuous'
        or 'categorical')
        :return: a list of training design matrices separated by class
        """

        features = self.continuous_features if feature_type == 'continuous' else self.categorical_features
        # Specify the structure of returned list
        class_count = np.bincount(y_train)
        X_train_classes = []
        for i in range(self.num_classes):
            X_train_classes.append(np.zeros(shape=(class_count[i], len(features))))
        # Record which row we are currently at for each class in the design matrix
        row_counter = np.zeros(shape=self.num_classes, dtype=int)
        for i in range(X_train.shape[0]):
            cur_class = y_train[i]
            X_train_classes[cur_class][row_counter[cur_class]] = X_train[i, features]
            row_counter[cur_class] += 1
        return X_train_classes

    def compute_gaussian_mean_var(self):
        """
        Compute mean and variance for all continuous data
        """

        # Finally, calculate mean and standard deviations
        for i in range(self.num_classes):
            for j in range(self.num_cont_features):
                self.gaussian_mean[i, j] = np.mean(self.X_train_classes_cont[i][:, j])
                self.gaussian_var[i, j] = np.var(self.X_train_classes_cont[i][:, j]) + self.alpha_cont

    def learn_continuous_likelihoods(self, x: np.ndarray):
        """
        Compute likelihoods of continuous features for every class

        :param x: a data point
        :return an array of likelihoods
        """

        cont_likelihoods = np.ones(shape=self.num_classes)
        for i in range(self.num_classes):
            likelihood = 1.0
            for j in range(self.num_cont_features):
                cur_value = x[self.continuous_features[j]]
                likelihood *= tools.gaussian_pdf(cur_value, self.gaussian_mean[i, j], self.gaussian_var[i, j])
            cont_likelihoods[i] = likelihood
        return cont_likelihoods

    def build_frequency_tables(self, X_train: np.ndarray):
        """
        Build frequency tables for every features, in which contains conditional probabilities of every
        category given some class

        :param X_train: the training design matrix
        """

        for i in range(self.num_cat_features):
            num_categories = np.bincount(X_train[:, self.categorical_features[i]].astype(int)).shape[0]
            self.frequency_tables[i] = np.zeros(shape=(num_categories, self.num_classes))
            for j in range(self.num_classes):
                # Count the occurrence time of each category
                category_count = np.zeros(shape=num_categories)
                for row_value in self.X_train_classes_cat[j][:, i].astype(int):
                    category_count[row_value] += 1
                # Compute the frequency for each category and class with smoothing
                self.frequency_tables[i][:, j] =\
                    (category_count + self.alpha_cat) / (category_count.sum() + self.alpha_cat * self.num_cat_features)

    def learn_categorical_likelihoods(self, x: np.ndarray):
        """
        Compute likelihoods of categorical features for every class

        :param x: a data point
        :return: an array of likelihoods
        """

        cat_likelihoods = np.ones(shape=self.num_classes)
        for i in range(self.num_classes):
            likelihood = 1.0
            for j in range(self.num_cat_features):
                cur_category = int(x[self.categorical_features[j]])
                # If there is an unseen category, compute with pure smoothing to avoid zero probabilities
                if cur_category >= self.frequency_tables[j].shape[0]:
                    likelihood *= (self.alpha_cat /
                                   (self.X_train_classes_cat[i].shape[0] + self.alpha_cat * self.num_cat_features))
                else:
                    likelihood *= self.frequency_tables[j][cur_category, i]
            cat_likelihoods[i] = likelihood
        return cat_likelihoods

    def learn_posterior(self, likelihoods: np.ndarray):
        """
        Compute (pseudo) posterior probabilities for every class. Notice that we do not account for the
        evidence term, because it has no effect on final results and can also save computation time

        :param likelihoods: the array of likelihoods of every class
        :return: an array of posterior probabilities
        """

        posteriors = np.zeros(shape=self.num_classes)
        for i in range(self.num_classes):
            posteriors[i] = likelihoods[i] * self.priors[i]
        return posteriors

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit a mixed Naive Bayes model given training data

        :param X_train: the training design matrix
        :param y_train: the array of training labels
        """

        # Learn class prior probabilities
        self.learn_priors(y_train)
        # Configure remaining attributes according to the training data
        num_features = X_train.shape[1]
        self.continuous_features = list(set(range(num_features)) - set(self.categorical_features))
        self.num_classes = len(self.priors)
        self.num_cont_features = len(self.continuous_features)
        self.gaussian_mean = np.zeros(shape=(self.num_classes, self.num_cont_features))
        self.gaussian_var = np.zeros(shape=(self.num_classes, self.num_cont_features))
        # Separate the training data by classes and by continuous / categorical features
        self.X_train_classes_cont = self.extract_features_by_class(X_train, y_train, feature_type='continuous')
        self.X_train_classes_cat = self.extract_features_by_class(X_train, y_train, feature_type='categorical')
        # Compute mean and variance for continuous data
        self.compute_gaussian_mean_var()
        # Build frequency tables for categorical data
        self.build_frequency_tables(X_train)

    def predict(self, X_test: np.ndarray):
        """
        Predict labels for unseen data using the trained model

        :param X_test: the test design matrix
        :return: an array of predicted labels
        """

        num_samples = X_test.shape[0]
        y_pred = np.zeros(shape=num_samples, dtype=int)
        for i in range(num_samples):
            cont_likelihoods = self.learn_continuous_likelihoods(X_test[i])
            cat_likelihoods = self.learn_categorical_likelihoods(X_test[i])
            posteriors = self.learn_posterior(cont_likelihoods * cat_likelihoods)
            # Choose the class with maximum posterior as our final prediction
            y_pred[i] = posteriors.argmax()
        return y_pred

from copy import deepcopy
from numpy import array, ndarray, sum
from sklearn.linear_model import LogisticRegression
from max_entropy.optimisation import get_max, get_max_o3
from max_entropy.optimisation_limited import get_max as get_max_lim, get_max_o3 as get_max_o3_lim
from max_entropy.optimisation_slack import get_max as get_max_slack, get_max_o3 as get_max_o3_slack
from max_entropy.poisson import fit_poisson, predict_poisson


class MultiLabelMaxEnt:

    def __init__(self, bin_cl=LogisticRegression(), var_limits=None, poisson_prediction=False, order=2, max_iter=100):
        """ :param bin_cl: a binary classifier, which must have a predict_proba function
            :param var_limits: None or a 2 element array with min and max number of labels allowed in the output vectors
            :param poisson_prediction: whether to predict the number of classes in the outputs beforehand
            :param order: the order of the statistics to consider (2 or 3)
            :param max_iter: the maximum number of iterations the optimisation solver may use
            After the fit function is called, the following variables will contain:
                - C: the number of labels
                - cl_a: a dictionary containing the classifiers to predict each individual label
                - cl_ab: a 2-level dictionary containing the classifier to predict pairs of labels
                - cl_abc: if order >= 3, a 3-level dictionary with classifiers to predict triplets of labels
                - poisson_prediction: if poisson_prediction, the poisson regression model built on training data
        """
        self.C = None
        self.bin_cl = bin_cl
        self.var_limits = var_limits
        self.poisson_prediction = poisson_prediction
        self.order = order
        self.max_iter = max_iter
        self.cl_a = {}
        self.cl_ab = {}
        if order >= 3:
            self.cl_abg = {}
        if poisson_prediction:
            self.poisson_mod = None


    def fit(self, X, Y):
        """ Fits the training data into the different classifiers, which will be stored in the
            cl_a and cl_ab (and if applicable, cl_abg) dictionaries
            If required, fits the Poisson regression algorithm, which will be stored in poisson_mod
            :param X: the feature matrix
            :param Y: the label matrix
        """
        if type(X) is not array and type(X) is not ndarray:
            X = array(X)
        if type(Y) is not array and type(Y) is not ndarray:
            Y = array(Y)

        self.C = C = len(Y[0])

        # Fit a classifier for each label
        for i in range(C):
            cl = deepcopy(self.bin_cl)
            cl.fit(X, Y[:, i])

            self.cl_a[i] = cl

        # Fit a classifier for each pair of labels
        for i, j in [(i, j) for i in range(C) for j in range(C) if i < j]:
            if sum(Y[:, i] * Y[:, j]) != 0:
                cl = deepcopy(self.bin_cl)
                cl.fit(X, Y[:, i] * Y[:, j])
            else:
                cl = 0

            if i not in self.cl_ab:
                self.cl_ab[i] = {}
            self.cl_ab[i][j] = cl

        # If needed, fits a classifier for each triplet of labels
        if self.order >= 3:
            for i, j, l in [(i, j, l) for i in range(C) for j in range(C) for l in range(C) if i < j < l]:
                if sum(Y[:, i] * Y[:, j] * Y[:, l]) != 0:
                    cl = deepcopy(self.bin_cl)
                    cl.fit(X, Y[:, i] * Y[:, j] * Y[:, l])
                else:
                    cl = 0

                if i not in self.cl_abg:
                    self.cl_abg[i] = {}
                if j not in self.cl_abg[i]:
                    self.cl_abg[i][j] = {}
                self.cl_abg[i][j][l] = cl

        # If needed, fits the Poisson regression
        if self.poisson_prediction:
            self.poisson_mod = fit_poisson(X, Y)


    def predict_example(self, x):
        """ Predicts the labels for one example using the maximum entropy method
            :param x: the feature vector of one example
            :return: the label vector predicted by MaxEnt for this example
        """
        if type(x) is not array and type(x) is not ndarray:
            x = [array(x)]
        else:
            x = [x]

        # Retrieve the probabilities for example x from the different classifiers
        f_a = {}
        f_ab = {}
        if self.order >= 3:
            f_abg = {}

        for i in range(self.C):
            f_a[i] = self.cl_a[i].predict_proba(x)[0][1]

        for i, j in [(i, j) for i in range(self.C) for j in range(self.C) if i < j]:
            if i not in f_ab:
                f_ab[i] = {}
            if self.cl_ab[i][j] == 0:
                f_ab[i][j] = 0
            else:
                f_ab[i][j] = self.cl_ab[i][j].predict_proba(x)[0][1]

        if self.order >= 3:
            for i, j, l in [(i, j, l) for i in range(self.C) for j in range(self.C) for l in range(self.C) if i < j < l]:
                if i not in f_abg:
                    f_abg[i] = {}
                if j not in f_abg[i]:
                    f_abg[i][j] = {}
                if self.cl_abg[i][j][l] == 0:
                    f_abg[i][j][l] = 0
                else:
                    f_abg[i][j][l] = self.cl_abg[i][j][l].predict_proba(x)[0][1]

        # Standard optimisation problem
        if self.var_limits is None and not self.poisson_prediction:
            if self.order == 2:
                return array(get_max(f_a, f_ab, self.C, self.max_iter))
            elif self.order == 3:
                return array(get_max_o3(f_a, f_ab, f_abg, self.C, self.max_iter))
        # Poisson regression and optimisation problem with slack variables
        elif self.poisson_prediction:
            n_labels = predict_poisson(x, self.poisson_mod)
            if n_labels > self.C:
                n_labels = self.C
            if self.order == 2:
                return array(get_max_slack(f_a, f_ab, self.C, n_labels, n_labels, self.max_iter))
            elif self.order == 3:
                return array(get_max_o3_slack(f_a, f_ab, f_abg, self.C, n_labels, n_labels, self.max_iter))
        # Optimisation problem with bounds on the number of labels in the output
        else:
            if self.order == 2:
                return array(get_max_lim(f_a, f_ab, self.C, self.var_limits[0], self.var_limits[1], self.max_iter))
            elif self.order == 3:
                return array(get_max_o3_lim(f_a, f_ab, f_abg, self.C, self.var_limits[0], self.var_limits[1], self.max_iter))


    def predict(self, X):
        """ Predicts the labels for each example using the maximum entropy method
            :param X: the features matrix
            :return: the label matrix predicted by MaxEnt
        """
        if type(X) is not array and type(X) is not ndarray:
            X = array(X)

        N = len(X)

        # Retrieve the probabilities for all the examples in X from the different classifiers
        fm_a = {}
        fm_ab = {}
        if self.order >= 3:
            fm_abg = {}

        for i in range(self.C):
            fm_a[i] = self.cl_a[i].predict_proba(X)[:, 1]

        for i, j in [(i, j) for i in range(self.C) for j in range(self.C) if i < j]:
            if i not in fm_ab:
                fm_ab[i] = {}
            if self.cl_ab[i][j] == 0:
                fm_ab[i][j] = array([0] * N)
            else:
                fm_ab[i][j] = self.cl_ab[i][j].predict_proba(X)[:, 1]

        if self.order >= 3:
            for i, j, l in [(i, j, l) for i in range(self.C) for j in range(self.C) for l in range(self.C) if i < j < l]:
                if i not in fm_abg:
                    fm_abg[i] = {}
                if j not in fm_abg[i]:
                    fm_abg[i][j] = {}
                if self.cl_abg[i][j][l] == 0:
                    fm_abg[i][j][l] = array([0] * N)
                else:
                    fm_abg[i][j][l] = self.cl_abg[i][j][l].predict_proba(X)[:, 1]

        # Iterate over the examples to perform the different optimisations
        Y = []
        for p in range(N):
            # Retrieve the probabilities for example x_p
            f_a = {}
            f_ab = {}
            if self.order >= 3:
                f_abg = {}

            for i in fm_a.keys():
                f_a[i] = fm_a[i][p]

            for i in fm_ab.keys():
                if i not in f_ab:
                    f_ab[i] = {}
                for j in fm_ab[i].keys():
                    f_ab[i][j] = fm_ab[i][j][p]

            if self.order >= 3:
                for i in fm_abg.keys():
                    if i not in f_abg:
                        f_abg[i] = {}
                    for j in fm_abg[i].keys():
                        if j not in f_abg[i]:
                            f_abg[i][j] = {}
                        for l in fm_abg[i][j].keys():
                            f_abg[i][j][l] = fm_abg[i][j][l][p]

            # Standard optimisation problem
            if self.var_limits is None and not self.poisson_prediction:
                if self.order == 2:
                    Y.append(array(get_max(f_a, f_ab, self.C, self.max_iter)))
                elif self.order == 3:
                    Y.append(array(get_max_o3(f_a, f_ab, f_abg, self.C, self.max_iter)))
            # Poisson regression and optimisation problem with slack variables
            elif self.poisson_prediction:
                n_labels = predict_poisson(X[p], self.poisson_mod)
                if n_labels > self.C:
                    n_labels = self.C
                if self.order == 2:
                    Y.append(array(get_max_slack(f_a, f_ab, self.C, n_labels, n_labels, self.max_iter)))
                elif self.order == 3:
                    Y.append(array(get_max_o3_slack(f_a, f_ab, f_abg, self.C, n_labels, n_labels, self.max_iter)))
            # Optimisation problem with bounds on the number of labels in the output
            else:
                if self.order == 2:
                    Y.append(array(get_max_lim(f_a, f_ab, self.C, self.var_limits[0], self.var_limits[1], self.max_iter)))
                elif self.order == 3:
                    Y.append(array(get_max_o3_lim(f_a, f_ab, f_abg, self.C, self.var_limits[0], self.var_limits[1], self.max_iter)))

        return array(Y)

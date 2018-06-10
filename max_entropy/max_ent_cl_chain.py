from max_entropy.multi_label_max_ent import MultiLabelMaxEnt
from sklearn.linear_model import LogisticRegression
from numpy import array, ndarray, concatenate, transpose
from copy import deepcopy


class MaxEntClChain:

    def __init__(self, bin_cl=LogisticRegression(), maxent_nr=5, label_order=None, max_iter=100):
        """ :param bin_cl: a binary classifier, which must have a predict_proba function
            :param maxent_nr: the number of labels to be predicted by the maximum entropy method
            :param label_order: an array containing the order of the constraints in the chain
                                (by default the order of the columns in the label matrix)
            :param max_iter: the maximum number of iterations the optimisation solver may use
            After the fit function is called, the following variables will contain:
                - C: the number of labels
                - maxent_mod: the trained maximum entropy classifier
                - chain: an array containing the different binary classifiers in the chain
        """
        self.C = None
        self.bin_cl = bin_cl
        self.maxent_nr = maxent_nr
        self.label_order = label_order
        self.maxent_mod = MultiLabelMaxEnt(bin_cl=bin_cl, max_iter=max_iter)
        self.chain = None


    def fit(self, X, Y):
        """ Fits the training data to MaxEnt and the classifiers in the chain
            :param X: the feature matrix
            :param Y: the label matrix
        """
        if type(X) is not array and type(X) is not ndarray:
            X = array(X)
        if type(Y) is not array and type(Y) is not ndarray:
            Y = array(Y)

        self.C = C = len(Y[0])
        if self.label_order is None:  # Set the default order
            self.label_order = [i for i in range(C)]

        # Train MaxEnt on the maxent_nr first constraints
        Y_maxent = Y[:, self.label_order[0 : self.maxent_nr]]
        self.maxent_mod.fit(X, Y_maxent)

        self.chain = []
        for nr in range(self.maxent_nr, C):
            # Train a binary classifier, with the preceding labels as additional features
            cl = deepcopy(self.bin_cl)
            X_cl = concatenate((X, Y[:, self.label_order[0 : nr]]), axis=1)
            Y_cl = Y[:, self.label_order[nr]]
            cl.fit(X_cl, Y_cl)
            self.chain.append(cl)


    def predict(self, X):
        """ Predicts the labels for each example
            :param X: the features matrix
            :return: the label matrix
        """
        Y = self.maxent_mod.predict(X)

        for cl in self.chain:
            X_cl = concatenate((X, Y), axis=1)
            y = cl.predict(X_cl)
            Y = concatenate((Y, transpose(array([y]))), axis=1)

        return Y

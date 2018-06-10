from numpy import sum, round as round_array
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Poisson


def fit_poisson(X, Y):
    """ Fits the Poisson regression model with the training data
        :param X: the feature matrix
        :param Y: the label matrix
        :return: the fitted Poisson model (instance of statsmodels.genmod.generalized_linear_model.GLMResults)
    """
    t = sum(Y, axis=1)
    pr = GLM(t, X, family=Poisson())
    return pr.fit()


def predict_poisson(x, mod):
    """ Predicts the expected number of labels in the output for one example
        :param x: the feature vector of the example to be predicted
        :param mod: the fitted poisson model (instance of statsmodels.genmod.generalized_linear_model.GLMResults)
        :return: the output of the Poisson regression for x, rounded to the nearest integer
    """
    return int(round_array(mod.predict(x)))

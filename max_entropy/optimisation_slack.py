from scipy.optimize import minimize
from numpy import array, argmax
from math import log
from itertools import product


def B(C, LB, UB):
    """ :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :return: returns a list containing all the binary vectors of length C
    """
    bin = []
    for k in product("01", repeat=C):
        vec = tuple(map(int, k))
        if LB <= sum(vec) and sum(vec) <= UB:
            bin.append(vec)
    return bin


def entropy(X, nb_var):
    """ The objective function (i.e. entropy measure)
        :param X: the 2^C variables and the slack variables
        :param nb_var: the number of probability variables (different from slack variables)
        :return: the entropy of the variables
    """
    ent = 0.0
    for k in range(nb_var):
        if X[k] != 0.0:
            ent += X[k] * log(X[k])

    slack = 0.0
    for s in range(nb_var, len(X)):
        slack += X[s]

    return ent + slack


def density(X, nb_var):
    """ The probability density constraint (sum P(k) = 1)
        :param X: the 2^C variables and the slack variables
        :param nb_var: the number of probability variables (different from slack variables)
        :return: the probability density constraint
    """
    cons = 0.0
    for k in range(nb_var):
        cons += X[k]
    return cons - 1


def order1(X, i, f, C, LB, UB, nb_var, ctr_nr):
    """ The first order statistic constraint for label i (sum k_i P(k) = f_a(i))
        :param X: the 2^C variables and the slack variables
        :param i: the label to consider
        :param f: the conditional probability of label i
        :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :param nb_var: the number of probability variables (different from slack variables)
        :param ctr_nr: the sequential number attributed to this constraint
        :return: the class probability constraint
    """
    cons = 0.0
    bin = B(C, LB, UB)
    for k in range(nb_var):
        cons += bin[k][i] * X[k]
    return cons - f + X[nb_var + ctr_nr] - X[nb_var + ctr_nr + 1]


def order2(X, i, j, f, C, LB, UB, nb_var, ctr_nr):
    """ The second order statistic constraint for labels i, j (sum k_i k_j P(k) = f_ab(i, j))
        :param X: the 2^C variables and the slack variables
        :param i: the two label
        :param j:   to consider
        :param f: the conditional probability of label i
        :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :param nb_var: the number of probability variables (different from slack variables)
        :param ctr_nr: the sequential number attributed to this constraint
        :return: the pairwise class probability constraint
    """
    cons = 0.0
    bin = B(C, LB, UB)
    for k in range(nb_var):
        cons += bin[k][i] * bin[k][j] * X[k]
    return cons - f + X[nb_var + ctr_nr] - X[nb_var + ctr_nr + 1]


def order3(X, i, j, l, f, C, LB, UB, nb_var, ctr_nr):
    """ The third order statistic constraint for label i, j, l (sum k_i k_j k_l P(k) = f_abg(i, j, l))
        :param X: the 2^C variables and the slack variables
        :param i: the three label
        :param j:   to consider
        :param l:
        :param f: the conditional probability of label i
        :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :param nb_var: the number of probability variables (different from slack variables)
        :param ctr_nr: the sequential number attributed to this constraint
        :return: the three-wise probability density constraint
    """
    cons = 0.0
    bin = B(C, LB, UB)
    for k in range(nb_var):
        cons += bin[k][i] * bin[k][j] * bin[k][l] * X[k]
    return cons - f + X[nb_var + ctr_nr] - X[nb_var + ctr_nr + 1]


def get_max(f_a, f_ab, C, LB, UB, max_iter):
    """ Performs the maximum entropy optimisation
        :param f_a: the conditional probabilities of the labels
        :param f_ab: the conditional probabilities of pairs of labels
        :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :param max_iter: the maximum number of iterations to perform during optimisation
        :return: the k vector with maximum probability
    """
    nb_var = len(B(C, LB, UB))

    cons = [{"type": "eq", "fun": density, "args": (nb_var,)}]

    ctr_nr = 0

    for i in range(C):
        cons.append({"type": "eq", "fun": order1, "args": (i, f_a[i], C, LB, UB, nb_var, ctr_nr)})
        ctr_nr += 2

    for i, j in [(i, j) for i in range(C) for j in range(C) if i < j]:
        cons.append({"type": "eq", "fun": order2, "args": (i, j, f_ab[i][j], C, LB, UB, nb_var, ctr_nr)})
        ctr_nr += 2

    nb_var_slack = nb_var + (len(cons) - 1) * 2

    res = minimize(fun=entropy, args=(nb_var,), x0=[0.2] * nb_var_slack, bounds=[(0, 1)] * nb_var_slack,
                   constraints=cons, method="SLSQP", options={"maxiter": max_iter, "disp": False})

    return B(C, LB, UB)[argmax(res.x[:nb_var])]


def get_max_o3(f_a, f_ab, f_abg, C, LB, UB, max_iter):
    """ Performs the maximum entropy optimisation with third order constraints
        :param f_a: the conditional probabilities of the labels
        :param f_ab: the conditional probabilities of pairs of labels
        :param f_abg: the conditional probabilities of triplets of labels
        :param C: the number of labels
        :param LB: the minimum number of positive labels authorised
        :param UB: the maximum number of positive labels authorised
        :param max_iter: the maximum number of iterations to perform during optimisation
        :return: the k vector with maximum probability
    """
    nb_var = len(B(C, LB, UB))

    cons = [{"type": "eq", "fun": density, "args": (nb_var,)}]

    ctr_nr = 0

    for i in range(C):
        cons.append({"type": "eq", "fun": order1, "args": (i, f_a[i], C, LB, UB, nb_var, ctr_nr)})
        ctr_nr += 2

    for i, j in [(i, j) for i in range(C) for j in range(C) if i < j]:
        cons.append({"type": "eq", "fun": order2, "args": (i, j, f_ab[i][j], C, LB, UB, nb_var, ctr_nr)})
        ctr_nr += 2

    for i, j, l in [(i, j, l) for i in range(C) for j in range(C) for l in range(C) if i < j < l]:
        cons.append({"type": "eq", "fun": order3, "args": (i, j, l, f_abg[i][j][l], C, LB, UB, nb_var, ctr_nr)})
        ctr_nr += 2

    nb_var_slack = nb_var + (len(cons) - 1) * 2

    res = minimize(fun=entropy, args=(nb_var,), x0=[0.2] * nb_var_slack, bounds=[(0, 1)] * nb_var_slack,
                   constraints=cons, method="SLSQP", options={"maxiter": max_iter, "disp": False})

    return B(C, LB, UB)[argmax(res.x[:nb_var])]

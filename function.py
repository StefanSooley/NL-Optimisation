import numpy as np


def f(x):
    """"
    :param x: the vector of the points x1, x2, x3
    :return: the value of the function f at points x1, x2, x3
    """
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2


def f_(x):
    """
    :param x: the vector of the points x1, x2, x3
    :return: the value of the gradient of f at points x1, x2, x3
    """
    return np.array([2.0 * x[0], 2.0 * x[1], 2.0 * x[2]])


Hessian = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

b = np.array([-2.0, -4.0, -6.0])

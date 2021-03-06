import numpy as np
import time
from methods import line_search


def f2(theta):
    x = np.array([[1, 25, 2], [1, 12, 42], [1, 11, 31], [1, 15, 35]])
    y = np.array([5, 25, 22, 18])
    return 1 / (2 * len(y)) * np.linalg.norm((y - x @ theta) ** 2)


def f2_(theta):
    x = np.array([[1, 25, 2], [1, 12, 42], [1, 11, 31], [1, 15, 35]])
    y = np.array([5, 25, 22, 18])
    deriv = 1 / (len(y)) * np.array((x @ theta - y)) @ x
    return deriv


def conjugate_gradient(x0, tol, f, f_, simple_logs=False):
    """
    Calculates the minimum value of a function using the Conjugate Gradient descent method.
    :param x0: Seed vector
    :param tol: precision of solution
    :param f: function to minimise
    :param f_: derivative function of f
    :param simple_logs: True if in depth logs are not required
    :return: solution vector, and logs.
    """
    # Start timing the algorithm
    t1 = time.perf_counter()

    # Calculate the initial g, and the first d is -g
    g = np.array(f_(x0))
    d = -g

    # Initialise the x vector
    x = x0

    # Steps counter
    j = 0

    # Initialise all the arrays to save intermediate values (for logs)
    alphas = np.array([])
    betas = np.array([])
    gs = np.array(g)
    g_olds = np.zeros(len(x0))
    ds = np.array(d)
    xs = np.array(x0)

    eps = np.inf
    while eps > tol:
        x_prev = x

        # Save old g for calculation of beta in next iteration
        g_old = g.copy()

        # In the first iteration, j=0 and d isn't adjusted
        if j == 0:
            d = d
        # Calculating beta using Polak-Ribiere form after the first iteration.
        else:

            # Calculate new g
            g = np.array(f_(x))
            beta = (g.transpose() @ (g - g_old) / (g_old.transpose() @ g_old))
            #beta = (g.transpose() @ g) / (g_old.transpose() @ g_old)
            betas = np.append(betas, beta)
            d = -g + beta * d

        # alpha calculated using a 1D line search algorithm (Golden section search)
        alpha = line_search(lambda y: f(x + y * d), -1, 1)

        # calculate new x
        x = x + alpha * d

        # Save values to logs
        alphas = np.append(alphas, alpha)
        xs = np.vstack([xs, x])
        ds = np.vstack([ds, d])
        g_olds = np.vstack([g_olds, g_old])
        gs = np.vstack([gs, g])

        eps = np.linalg.norm(x - x_prev)

        j += 1

    # Take the time after the loop (difference between t2 and t1 is the total time taken)
    t2 = time.perf_counter()
    if simple_logs:
        logs = np.array([j, t2 - t1, xs], dtype='object')
    else:
        logs = np.array([j, xs, ds, gs, alphas, betas, t2 - t1], dtype='object')

    return x, logs


def steepest_descent(x0, tol, f, f_):
    """
    Calculates the minimum value of a function using the Steepest descent method.
    :param x0: Seed vector
    :param tol: precision of solution
    :param f: function to minimise
    :param f_: derivative function of f
    :return: solution vector, and logs.
    """
    # Start timing the algorithm

    t1 = time.perf_counter()

    eps = np.inf
    x = x0
    j = 0
    xs = np.array(x0)

    while eps > tol:
        if j == 0:
            g = -np.array(f_(x0))
        else:
            g = -np.array(f_(x))
        alpha = line_search(lambda y: f(x + y * g), -1, 1)

        x_prev = x

        x = x + alpha * g
        eps = np.linalg.norm(x - x_prev)
        j += 1
        xs = np.vstack([xs, x])

    # Take the time after the loop (difference between t2 and t1 is the total time taken)
    t2 = time.perf_counter()

    logs = np.array([j, t2 - t1, xs], dtype='object')

    return x, logs


def min_cost_func(theta0, tol, f, f_):
    # x = np.array([[1, 25, 2], [1, 12, 42], [1, 11, 31], [1, 15, 35]])
    # y = np.array([5, 25, 22, 18])
    # print(np.linalg.inv(x.transpose()@x)@x.transpose()@y)

    sol, logs = conjugate_gradient(theta0, tol, f, f_)
    #sol, logs = steepest_descent(theta0, tol, f, f_)
    return sol, logs

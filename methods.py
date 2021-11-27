import numpy as np
import time
import pandas as pd

golden_ratio = (np.sqrt(5) + 1) / 2


def line_search(function, start, end):
    # Implementation of the Golden section search algorithm.
    A = end - (end - start) / golden_ratio
    B = start + (end - start) / golden_ratio
    while np.abs(B - A) > 10 ** -6:
        if function(A) < function(B):
            end = B
        else:
            start = A
        A = end - (end - start) / golden_ratio
        B = start + (end - start) / golden_ratio

    return (B + A) / 2


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
    g = np.array(f_(*x0))
    d = -g

    # Initialise the x vector
    x = np.zeros(len(x0))

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
            g = np.array(f_(*x))

            print(g, g_old)

            beta = (g.transpose() @ (g - g_old) / (g_old.transpose() @ g_old))
            beta = (g.transpose()@g) / (g_old.transpose() @ g_old)
            betas = np.append(betas, beta)
            d = -g + beta * d

        # alpha calculated using a 1D line search algorithm (Golden section search)
        alpha = line_search(lambda y: f(*x + y * d), -1, 1)

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
    # Start timing the algorithm
    t1 = time.perf_counter()

    eps = np.inf
    x = np.zeros(len(x0))
    j = 0
    xs = np.array(x0)

    while eps > tol:
        if j == 0:
            g = -np.array(f_(*x0))
        else:
            g = -np.array(f_(*x))
        alpha = line_search(lambda y: f(*x + y * g), -1, 1)

        x_prev = x

        x = x + alpha * g
        eps = np.linalg.norm(x - x_prev)
        j += 1
        xs = np.vstack([xs, x])

    # Take the time after the loop (difference between t2 and t1 is the total time taken)
    t2 = time.perf_counter()

    logs = np.array([j, t2-t1, xs], dtype='object')


    return x, logs

def min_cost_func(x0, tol, f, f_):
    #data = pd.read_csv('fires.csv')

    x = np.array([[1, 25, 2], [1, 12, 42], [1, 11, 31], [1, 15, 35]])

    y = np.array([5, 25, 22, 18])
    theta = np.array([1, 0, 0.5])

    print(np.linalg.norm((1/4)*(y-x@theta)**2))

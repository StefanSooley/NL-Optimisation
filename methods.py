import numpy as np
import time
from function import Hessian, b, f, f_

golden_ratio = (np.sqrt(5) + 1) / 2


def line_search(function, start, end):
    # Implementation of the Golden section search algorithm.
    A = end - (end - start) / golden_ratio
    B = start + (end - start) / golden_ratio
    while abs(B - A) > 10 ** -6:
        if function(A) < function(B):
            end = B
        else:
            start = A
        A = end - (end - start) / golden_ratio
        B = start + (end - start) / golden_ratio

    return (B + A) / 2


def conjugate_gradient(x0, tol, return_logs=True):
    # Start timing the algorithm
    t1 = time.time()

    # Calculate the initial g, and the first d is -g
    g = Hessian @ x0 + b
    d = -g

    # Initialise the x vector
    x = np.zeros(3)

    # Steps counter
    j = 0

    # Initialise all the arrays to save intermediate values (for logs)
    alphas = np.array([])
    betas = np.array([])
    gs = np.array(g)
    g_olds = np.array([0, 0, 0])
    ds = np.array(d)
    xs = np.array(x0)

    while np.linalg.norm(d) > tol:
        # In the first iteration, j=0 and d isn't adjusted
        if j == 0:
            d = d
        # Calculating beta using Polak-Ribiere form after the first iteration.
        else:
            beta = (g.transpose() @ (g - g_old) / (g_old.transpose() @ g_old))
            betas = np.append(betas, beta)
            d = -g + beta * d

        # alpha calculated using a 1D line search algorithm (Golden section search)
        alpha = line_search(lambda y: f(x + y * d), 0, 1)

        # calculate new x
        x = x + alpha * d

        # Save old g for calculation of beta in next iteration
        g_old = g.copy()

        # Calculate new g
        g = Hessian @ x + b

        # Save values to logs
        alphas = np.append(alphas, alpha)
        xs = np.vstack([xs, x])
        ds = np.vstack([ds, d])
        g_olds = np.vstack([g_olds, g_old])
        gs = np.vstack([gs, g])

        j += 1

    # Take the time after the loop (difference between t2 and t1 is the total time taken)
    t2 = time.time()

    logs = np.array([j, xs, ds, gs, alphas, betas, t2 - t1], dtype='object')

    return x, logs

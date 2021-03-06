import numpy as np
import time
from methods import conjugate_gradient, steepest_descent
from regression import min_cost_func, f2, f2_
from data import save_logs, read_func
from plot import plot_contour


def main():
    f, f_ = read_func('input.txt')

    simple_logs = True
    tolerance = 10 ** -5
    x0 = np.array([-5, -5])
    theta0 = np.array([1, 1, 1])

    sol, logs = min_cost_func(theta0, tolerance, f2, f2_)
    print(sol, logs[0])

    # solution_conjg, logs1 = conjugate_gradient(x0, tolerance, f, f_, simple_logs)
    # solution_steep, logs2 = steepest_descent(x0, tolerance, f, f_)
    #
    # print(f"Solution = {solution_steep}, {solution_conjg}")
    #
    # if not simple_logs:
    #     save_logs(logs1, filename='output.txt')
    #
    # print(f"Conjugate Gradient: {logs1[0:2]}\nSteepest Descent: {logs2[0:2]}")
    # plot_contour(f, logs1[2], logs2[2])


if __name__ == '__main__':
    main()

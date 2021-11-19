import numpy as np
import time
from methods import conjugate_gradient
from data import save_logs, read_func


def main():
    read_func('input.txt')

    # tolerance = 10 ** -6
    # x0 = np.array([0.5, 0.5, 0.5])
    #
    # solution, logs = conjugate_gradient(x0, tolerance)
    # print(f"Solution = {solution}")
    # save_logs(logs, filename='output.txt')


if __name__ == '__main__':
    main()

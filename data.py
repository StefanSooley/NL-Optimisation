import numpy as np
import sympy


def read_func(input_filename):
    """
    :param input_filename: The filename containing the input function

    Reads the input file and parses to produce a function.
    :return: The function as a python function, and it's derivative.
    """
    with open(input_filename) as f:
        lines = f.readlines()

    # Reads the function on the first line
    function = lines[0]

    # Converts it to a sympy object
    func = sympy.sympify(function)

    # Make a string of the list of variables, remove commas and brackets
    syms_str = str(sorted(func.free_symbols, key=lambda s: s.name)).replace(",", '')[1:-1]

    # Convert the string of variables into sympy variables
    symbols = sympy.symbols(syms_str)

    # Calculate grad, and turn it into a python function
    f_ = sympy.lambdify(symbols, np.array([func.diff(x) for x in symbols]))

    # Convert the expression into a python function
    f = sympy.lambdify(symbols, func)

    return f, f_


def save_logs(logs, filename='outs.txt'):
    """
    Takes the logs from the algorithm and writes an output file with
    :param logs:
    :param filename:
    :return:
    """

    j, xs, ds, gs, alphas, betas, time = logs
    steps_msg = ""

    msg1 = f"====== ====== Overview ====== ======\n\n" \
           f"Number of iterations: {j}\n" \
           f"Solution: {xs[-1]}\n" \
           f"Time taken: {round(time * 10 ** 3, 3)} ms\n\n"

    msg2 = f"======= ====== Steps ====== =======\n\n" \
           f"To calculate each successive x vector iteratively, the formula: x_new = x + alpha * d is used.\nThis " \
           f"will continue until the norm of d is less than a tolerance.\n" \
           f"Each d is calculated using: d_new = beta * d - g,\n" \
           f"where beta is calculated using the Polak-Ribiere formula.\n" \
           f"Alpha is approximated using a line search algorithm to minimise the function f(x + alpha*d). \n\n" \
           f"\n1)\nIn the first step, the seed vector x0 was {xs[0]}, and d was simply the negative of the gradient\n" \
           f"resulting in d = {ds[0]}. The new x vector is given by the equation x = x + alpha * d, " \
           f"\n\nThe resulting x vector is {xs[1]}.\n"

    for i in range(2, j + 1):
        if i < j:

            steps_msg += f"\n\n{str(i)})\nThe calculated beta value is: {betas[i - 2]}," \
                         f"\nmaking d {ds[i]}." \
                         f"\nThe alpha value found using line search minimisation is: {alphas[i - 2]}" \
                         f"\n\nUsing these values, the resulting x vector is {xs[i]}\n" \
                         f"\nSince the calculated norm of d is higher than the tolerance, another iteration is " \
                         f"required."
        else:
            steps_msg += f"\n\n{str(i)})\nThe calculated beta value is: {betas[i - 2]}," \
                         f"\nmaking d {ds[i]}." \
                         f"\nThe alpha value found using line search minimisation is: {alphas[-1]}" \
                         f"\n\nUsing these values, the resulting x vector is {xs[i]}\n" \
                         f"\nSince the calculated norm of d is now below the tolerance, " \
                         f"this x is the solution to the desired accuracy."
    out = msg1 + msg2 + steps_msg

    np.savetxt(filename, [out], delimiter=" ", fmt="%s")

    return None

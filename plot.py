import matplotlib.pyplot as plt
import numpy as np


def plot_contour(f, x1_arrs, x2_arrs):
    x1 = np.linspace(-5, 5, 100)  # x
    x2 = np.linspace(-5, 5, 100)  # y

    x_arrs = np.array([x1_arrs, x2_arrs], dtype='object')
    X1, X2 = np.meshgrid(x1, x2)
    Y = f(X1, X2)

    cs = ['purple', 'g']

    plt.rcParams["figure.figsize"] = (10, 7)

    cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(cp, inline=1, fontsize=10)
    cm = plt.cm.get_cmap('hot')
    plt.scatter(X1, X2, c=Y, cmap=cm)

    for c, x in enumerate(x_arrs):
        for i in range(len(x) - 1):
            plt.annotate('', xy=x[i + 1, :], xytext=x[i, :],
                         arrowprops={'arrowstyle': '->', 'color': cs[c], 'lw': 1.5},
                         va='center', ha='center')


    plt.text(4, -4, "Conjugate Gradient", size=10,
             ha="center", va="center", color='purple',
             bbox=dict(boxstyle="round",
                       ec=(0, 0, 0),
                       fc=(1., 1, 1),
                       )
             )
    plt.text(4, -4.5, "  Steepest Descent ", size=10,
             ha="center", va="center", color='green',
             bbox=dict(boxstyle="round",
                       ec=(0, 0, 0),
                       fc=(1., 1, 1),
                       )
             )

    plt.title('Plot of the paths taken by the different Gradient Descent methods')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    # cm = plt.cm.get_cmap('viridis')
    # plt.scatter(X1, X2, c=Y, cmap=cm)
    # plt.show()

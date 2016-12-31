import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


def random_data():
    """
    create a random data set
    :return: x and y
    """
    np.random.seed(0)
    return sklearn.datasets.make_moons(200, noise=0.20)


def show_data(x, y):
    """
    :param x:
    :param y:
    """
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y)
    plt.show()


if __name__ == "__main__":
    X, Y = random_data()
    show_data(X, Y)

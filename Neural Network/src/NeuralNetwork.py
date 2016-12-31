import numpy as np
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


def random_data():
    """
    create a random data set
    :return: x and y
    """
    np.random.seed(0)
    return sklearn.datasets.make_moons(200, noise=0.20)


def plot_decision_boundary(pred_func, x, y):

    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, z)
    plt.scatter(X[:, 0], x[:, 1], c=y)
    plt.show()


if __name__ == "__main__":
    X, Y = random_data()

    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, Y)

    plot_decision_boundary(lambda x: clf.predict(x), X, Y)

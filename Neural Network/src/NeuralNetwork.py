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
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


def calculate_loss(model, x, y):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Forward propagation to calculate the predictions
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #Calculating the loss
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    #Add the regularization term for the loss
    data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return 1./num_examples * data_loss

def predict(model, x):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Forward propagation
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(nn_hdim, x, y, num_passes=20000, print_loss=False):

    #Initialize the parameters to random values.
    np.random.seed(0)
    w1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # The model
    model = {}

    # Gradient Descent
    for i in range(1, num_passes):

        #Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        #Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        #add regularization terms
        dw2 += reg_lambda * w2
        dw1 += reg_lambda * w1

        #update gradient descent parameter
        w1 += -epsilon * dw1
        w2 += -epsilon * dw2
        b1 += -epsilon * db1
        b2 += -epsilon * db2

        #assign new parameters to the model
        model = {'w1' : w1, 'w2' : w2, 'b1': b1, 'b2' : b2}

        #print the loss(optional)
        if print_loss and i % 1000 == 0:
            print "loss after iteration %i : %f" % (i, calculate_loss(model, x, y))

    return model

if __name__ == "__main__":

    X, Y = random_data()
    num_examples = len(X)
    nn_input_dim = 2
    nn_output_dim = 2

    epsilon = 0.01
    reg_lambda = 0.01    

    model = build_model(3, X, Y, print_loss=False)
    plot_decision_boundary(lambda x : predict(model, x), X, Y)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show()

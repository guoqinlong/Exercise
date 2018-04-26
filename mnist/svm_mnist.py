import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plot

def read_data(filepath='digit-recognizer/train.csv', is_train=True):
    data = np.asarray(pd.read_csv(filepath, header=1))
    if is_train:
        x = data[:,1:]
        y = data[:,:1]
        return x, y
    else:
        return data


def pre_process(x):
    x[x > 0] = 1

def show_img(img, title='title'):
    img = img.reshape(28, 28)
    plot.imshow(img, cmap='gray')
    plot.title(title)
    plot.show()


def train():
    clf = SVC()
    x_train, y_train = read_data()
    show_img(x_train[33], 33)
    pre_process(x_train)
    # x_train, x_test, y_train, y_test = train_test_split(x, y)
    # clf.fit(x_train, y_train)
    # print(clf.score(x_test, y_test))
    return clf


def predict(clf):
    data = read_data("digit-recognizer/test.csv", False)
    pre_process(data)
    results = clf.predict(data)
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = "ImageId"
    df.columns = ['Label']
    df.to_csv('digit-recognizer/results.csv', header=True)

if __name__ == '__main__':
    clf = train()
    predict(clf)
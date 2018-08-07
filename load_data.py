import numpy as np
from mnist import MNIST

W = 28
H = 28

def load():
    mndata = MNIST("dataset/mnist")

    train_images, train_labels = mndata.load_training()
    train_labels = [vectorize(i) for i in train_labels]
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_images = np.reshape(train_images[0:1000,:], [-1, W, H, 1])
    train_labels = np.reshape(train_labels[0:1000,:], [-1, 10])

    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images)
    test_labels = [vectorize(i) for i in test_labels]
    test_labels = np.array(test_labels)
    test_images = np.reshape(test_images[0:100,:], [100, W, H, 1])
    test_labels = np.reshape(test_labels[0:100,:], [100, 10])

    train = (train_images, train_labels)
    test = (test_images, test_labels)

    return (train, test)

def vectorize(i):
    x = np.zeros((10,1))
    x[i] = 1
    return x

if __name__ == "__main__":
    load()

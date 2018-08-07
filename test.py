import numpy as np

def batch_shuffle(X, Y, batch_size, seed):

    [m, n_H, n_W, c] = X.shape

    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    number_of_batches = int(m / batch_size)

    for i in range(number_of_batches):
        batch_X = shuffled_X[i*batch_size : i*batch_size + batch_size]
        batch_Y = shuffled_Y[i*batch_size : i*batch_size + batch_size]
        mini_batches.append((batch_X, batch_Y))

    if m % batch_size != 0:
        batch_X_final = shuffled_X[number_of_batches*batch_size : m]
        batch_Y_final = shuffled_Y[number_of_batches*batch_size : m]
        mini_batches.append((batch_X_final, batch_Y_final))

    return mini_batches

X = np.random.randn(1000, 34, 34, 3)
Y = np.random.randn(1000,1)
batch_size = 64
seed = 0
re = batch_shuffle(X, Y, batch_size, seed)

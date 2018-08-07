import numpy as np
import tensorflow as tf
import load_data

# Convolutional layer 1
filter_size_1 = 5
number_of_filters_1 = 16


# Convolutional layer 2
filter_size_2 = 5
number_of_filters_2 = 36

# Fully-connected layer 1
fc_size = 128

# Fully-connected layer 2
number_classes = 10

def batch_shuffle(X, Y, batch_size, seed):

    [m, n_H, n_W, c] = X.shape

    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    number_of_batches = int(m / batch_size)

    for i in range(number_of_batches):
        batch_X = shuffled_X[i*batch_size : i*batch_size + batch_size,:,:,:]
        batch_Y = shuffled_Y[i*batch_size : i*batch_size + batch_size,:]
        mini_batches.append((batch_X, batch_Y))

    if m % batch_size != 0:
        batch_X_final = shuffled_X[number_of_batches*batch_size : m,:,:,:]
        batch_Y_final = shuffled_Y[number_of_batches*batch_size : m,:]
        mini_batches.append((batch_X_final, batch_Y_final))

    return mini_batches


def parameters_initializer(layer_number, weight_shape, bias_length):
    W = tf.get_variable("W" + str(layer_number), shape = weight_shape, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b" + str(layer_number), shape = bias_length, dtype=tf.float32)

    return W, b

def placeholder_initializer(width, height, channel, n_class):
    X = tf.placeholder(tf.float32, shape = [None, width, height, channel], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, n_class], name = "Y")

    return X, Y

def conv_layer(layer_number, prev_layer, filter_size, num_filters):
    prev_layer_channels = prev_layer.shape[3]
    weight_shape = [filter_size, filter_size, prev_layer_channels, num_filters]
    bias_length = [num_filters]

    W, b = parameters_initializer(layer_number, weight_shape, bias_length)
    # conv layer 
    Z = tf.nn.conv2d(input = prev_layer, filter = W, strides = [1,1,1,1], padding = "SAME")
    # add bias
    Z = Z + b
    # Relu
    A = tf.nn.relu(Z)
    # Max pool
    P = tf.nn.max_pool(value = A, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    return P, W

def fc_layer(layer_number, prev_layer, number_output, use_relu = True):
    weight_shape = [number_output, prev_layer.shape[1]]
    bias_shape = [1, number_output]

    W, b = parameters_initializer(layer_number, weight_shape, bias_shape)

    Z = tf.add(tf.matmul(prev_layer, tf.transpose(W)), b)

    A = tf.nn.relu(Z)
    return A

def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
    return cost

def forward_propagation(X):
    # first convolutional + Max Pooling
    A1, W1 = conv_layer(1, X, filter_size_1, number_of_filters_1)
    
    # second convolutional + Max pooling
    A2, W2 = conv_layer(2, A1, filter_size_2, number_of_filters_2)

    # flatten
    P2 = tf.contrib.layers.flatten(A2)
    
    # fully connected layer 1
    A3 = fc_layer(3, P2, fc_size, use_relu = True)
    
    # fully connected layer 2
    A4 = fc_layer(4, A3, number_classes, use_relu = True)

    return A4

def cnn_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.01, epoches = 10000, mini_batch_size = 64):
    
    [m, width, height, channels] = X_train.shape
    X, Y = placeholder_initializer(width, height, channels, 10)

    Z = forward_propagation(X)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # predicted class
    y_pred = tf.nn.softmax(Z)
    y_pred_class = tf.argmax(y_pred, axis = 1)
    y_true_class = tf.argmax(Y, axis = 1)
    correct_prediction = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        seed = 0

        for epoch in range(epoches):
            minibatch_cost = 0
            num_minibatches = int(m / mini_batch_size)
            seed = seed + 1
            mini_batches = batch_shuffle(X_train, Y_train, mini_batch_size, seed)

            for batch in mini_batches:
                X_batch, Y_batch = batch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: X_batch, Y: Y_batch})
                acc = sess.run(accuracy, feed_dict = {X:X_train, Y:Y_train})

                minibatch_cost += temp_cost / num_minibatches

            if epoch % 1 == 0:
                print("Cost after epoch %i: %f ---- accuracy: %f" %(epoch, minibatch_cost, acc))

if __name__ == "__main__":
    data = load_data.load()
    train, test = data
    X_train, Y_train = train
    X_test, Y_test = test
    cnn_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.01, epoches = 10000, mini_batch_size = 64)

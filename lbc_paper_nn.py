import numpy as np
import tensorflow as tf
import time
import sys

from utils.utils import batch_shuffle
from lbcnn.lbc_utils import generate_lbc_weights

from dataset.mnist_load import mnist_load
from dataset.cifar_load import cifar_load
from dataset.svhn_load import svhn_load


if './' not in sys.path:
    sys.path.append('./')


def placeholder_initializer(height, width, channel, n_class):
    """
    Arguments:
        height: height of input X
        width: width of input X
        channel: channel of input X
        n_class: number of class needed to classify at the end of CNN
    """
    X = tf.placeholder(tf.float32, shape = [None, height, width, channel], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, n_class], name = "Y")
    Keep_prob = tf.placeholder(tf.float32, name = "KeepProb")

    return X, Y, Keep_prob

def lbcnn(prev_input, layer_no, lbc_size, lbc_channels, output_channels, sparsity):
    """
    Arguments:
        prev_input: previous input X[l-1]
        layer_no: this is the number to assign into tf.get_variable of learnable
        lbc_size: the size of anchor weight, height = width = weight_size
        lbc_channels: total number of anchor weights
        output_channels: this will be the number of 1x1 convolution, therefore, it will be the number of output size
        sparsity: the sparsity of layer, value from 0 to 1
        isStochastic: True then each layer later, we will increase sparsity as suggested in the original paper
    """
    # the first part of lbcnn is fixed convolution through anchor weights

    # first, we need to initialize anchor weight
    prev_channels = prev_input.shape[3]

    weight = generate_lbc_weights(layer_no = layer_no, height = lbc_size, width = lbc_size, prev_output_channels = prev_channels, number_of_filters = lbc_channels, sparsity = sparsity)

    # batch normalization
    B = tf.contrib.layers.batch_norm(prev_input)

    Z = tf.nn.conv2d(input = B, filter = weight, strides = [1, 1, 1, 1], padding = "SAME")

    A1 = tf.nn.relu(Z)
    
    # the second step is 1x1 convolution

    I = tf.get_variable("I" + str(layer_no), shape = [1, 1, lbc_channels, output_channels], dtype=tf.float32)

    A2 = tf.nn.conv2d(input = A1, filter = I, strides = [1, 1, 1, 1], padding = "SAME")

    A2 = tf.add(A2, prev_input)

    return A2

def model(X, Keep_prob, options):

    ### parameters
    conv_layers = options['CONV_LAYERS']
    lbc_filters = options['LBC_FILTERS']
    lbc_size = options['LBC_SIZE']
    identity_filters = options['IDENTITY_FILTERS']
    fc_hidden_units = options['FC_HIDDEN_UNITS']
    output_classes = options['OUTPUT_CLASSES']
    sparsity = options['SPARSITY']
    ### 

    # convolutional layers
    # prev-convolution
    with tf.name_scope("Pre-Conv"):
        X1 = tf.contrib.layers.conv2d(inputs = X, num_outputs = identity_filters, kernel_size = lbc_size, stride = 1, padding = "SAME", activation_fn = None)
        X2 = tf.contrib.layers.batch_norm(X1)
        X3 = tf.nn.relu(X2)

    X_in = X3
    for i in range(conv_layers):
        with tf.name_scope("Lbcnn"):
            X_new = lbcnn(prev_input = X_in, layer_no = i, lbc_size = lbc_size, lbc_channels = lbc_filters, output_channels = identity_filters, sparsity = sparsity)
            X_in = X_new

    # average pool
    with tf.name_scope("AvgPool"):
        Z = tf.nn.avg_pool(value = X_in, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = "SAME")

    # flatter
    with tf.name_scope("Flatter"):
        P = tf.contrib.layers.flatten(Z)

    # add dropout
    with tf.name_scope("Dropout"):
        F1 = tf.nn.dropout(x = P, keep_prob = Keep_prob)

    # fully connected 1
    with tf.name_scope("Fully"):
        F1 = tf.contrib.layers.fully_connected(inputs = F1, num_outputs = fc_hidden_units, activation_fn = tf.nn.relu)

    # add dropout
    with tf.name_scope("Dropout"):
        F2 = tf.nn.dropout(x = F1, keep_prob = Keep_prob)

    # fully connected 2
    with tf.name_scope("Fully"):
        F2 = tf.contrib.layers.fully_connected(inputs = F2, num_outputs = output_classes, activation_fn = None)

    return F2

def compute_cost(A, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = A))

def CNN_training(X_train, Y_train, X_test, Y_test, options):

    ### parameters
    learning_rate = options['LEARNING_RATE']
    epoches = options['EPOCHES']
    batch_size = options['BATCH_SIZE']
    output_classes = options['OUTPUT_CLASSES']
    keep_prob = options['KEEP_PROB']
    ### 
    
    m, height, width, input_channels = X_train.shape

    X, Y, Keep_prob = placeholder_initializer(height, width, input_channels, output_classes)

    A = model(X, Keep_prob = Keep_prob, options = options)

    #print('----------CNN_training-------------')
    
    with tf.name_scope("Cost"):
        cost = compute_cost(A, Y)

    optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9).minimize(cost)

    # accuracy
    with tf.name_scope("Accuracy"):
        y_pred = tf.nn.softmax(A)
        y_pred_class = tf.argmax(y_pred, axis = 1)
        y_true_class = tf.argmax(Y, axis = 1)
        y_acc = tf.equal(y_true_class, y_pred_class)
        accuracy = tf.reduce_mean(tf.cast(y_acc, tf.float32))
    
    init = tf.global_variables_initializer()

    result_output = open("./result/result.txt", "a")

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graph/", sess.graph)
        sess.run(init)
        seed = 0
        for i in range(epoches):
            seed += 1
            mini_batches = batch_shuffle(X_train, Y_train, batch_size, seed = seed)
            mini_batch_cost = 0
            num_minibatches = int(m / batch_size) + int(m % batch_size != 0)
            batch_count = 0 
            for mini_batch in mini_batches:
                X_batch, Y_batch = mini_batch

                batch_count += 1
                start_time = time.time()

                _, opti_cost = sess.run([optimizer, cost], feed_dict = {X: X_batch, Y: Y_batch, Keep_prob: keep_prob})

                mini_batch_cost += opti_cost / num_minibatches
                end_time = time.time()

                print("|Epoches:  [%d][%d|%d]  Cost: %f   Time: %f" %(i, batch_count+1, num_minibatches, opti_cost,end_time - start_time))

            # test
            total_test_batches = int(X_test.shape[0]/batch_size)
            accumu_acc= 0
            for test_batch in range(total_test_batches):
                acc = sess.run(accuracy, feed_dict = {X: X_test[test_batch*batch_size:test_batch*batch_size+batch_size,:], Y: Y_test[test_batch*batch_size:test_batch*batch_size+batch_size,:], Keep_prob: 1})
                print("|Test:  [%d][%d|%d] Accuracy: %f\n"  %(i, test_batch+1, total_test_batches, acc))
                accumu_acc += acc

                result_output.write("|Test:  [%d][%d|%d] Accuracy: %f\n" %(i, test_batch, num_minibatches, acc))

            print("Total Accuracy: %f" %(accumu_acc/total_test_batches))
        writer.close()

def training(options):
    # mnist_load
    X1_train, Y1_train, X1_test, Y1_test = mnist_load()
    # cifar load
    X2_train, Y2_train, X2_test, Y2_test = cifar_load()
    # svhn
    X3_train, Y3_train, X3_test, Y3_test = svhn_load()

    # parameter initialization
    if options['DATASET'] == "mnist":
        X_train, Y_train, X_test, Y_test = X1_train, Y1_train, X1_test, Y1_test
    elif options['DATASET'] == "cifar":
        X_train, Y_train, X_test, Y_test = X2_train, Y2_train, X2_test, Y2_test

    CNN_training(X_train, Y_train, X_test, Y_test, options)

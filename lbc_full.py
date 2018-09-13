import numpy as np
import tensorflow as tf

from dataset.mnist_load import mnist_load

def batch_shuffle(X, Y, batch_size):

    [m, n_H, n_W, c] = X.shape

    mini_batches = []

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

def generate_lbc_weights(height, width, prev_output_channels, number_of_filters, sparsity):
    """
    Arguments:
        height: height of the filter
        width: width of the filter
        prev_output_channels: channels of previous output layers
        number_of_filters: total number of filters (channels)
        sparsity: from 0 to 1
    """
    # first step is based on sparsity levels, generate zero and non-zero(1)
    number_elements = height * width * prev_output_channels * number_of_filters
    w = np.random.rand(number_elements,1) <= sparsity
    w = w.astype(np.float32)
    
    # we now multiply each 1 in original w with each number 1 or -1 in bnl
    for i in range(w.shape[0]):
        if w[i] == 1:
            w[i] = w[i] * (np.random.binomial(1, 0.5, None)*2-1)
    
    # reshape and assign to a constant Tensor
    w = np.reshape(w, (height, width, prev_output_channels, number_of_filters))

    out = tf.constant(value = w, dtype = tf.float32)

    return out

def lbcnn(prev_input, lbc_size, lbc_channels, output_channels, sparsity):
    """
    Arguments:
        prev_input: previous input X[l-1]
        lbc_size: the size of anchor weight, height = width = weight_size
        lbc_channels: total number of anchor weights
        output_channels: number of 1x1 filters
        sparsity: the sparsity of layer, value from 0 to 1
    """
    # the first part of lbcnn is fixed convolution through anchor weights

    # first, we need to initialize anchor weight
    prev_channels = prev_input.shape[3]
    weight = generate_lbc_weights(height = lbc_size, width = lbc_size, \
        prev_output_channels = prev_channels, number_of_filters = lbc_channels, sparsity = sparsity)

    # batch normalization
    B = tf.contrib.layers.batch_norm(prev_input)
    
    # LBCNN conv
    Z = tf.nn.conv2d(input = B, filter = weight, strides = [1, 1, 1, 1], padding = "SAME")
    A1 = tf.nn.relu(Z)
    
    # the second step is 1x1 convolution
    A2 = tf.contrib.layers.conv2d(inputs = A1, num_outputs = output_channels, kernel_size = 1, \
            stride = 1, padding = "SAME", activation_fn = None)
    
    # add input and output, like Resnet
    A2 = tf.add(A2, prev_input)

    return A2

def model(X, Keep_probability, options):

    ### parameters for MNIST
    conv_layers = options['CONV_LAYERS'] #75
    lbc_filters = options['LBC_FILTERS'] #512
    lbc_size = options['LBC_SIZE'] #3
    output_channels = options['IDENTITY_FILTERS'] #16
    fc_hidden_units = options['FC_HIDDEN_UNITS'] #128
    output_classes = options['OUTPUT_CLASSES'] #10
    sparsity = options['SPARSITY'] #0.5
    ### 

    # prev-convolution so when add in+out like ResNet, we do not need to padding the shortcut
    with tf.name_scope("Pre-Conv"):
        X1 = tf.contrib.layers.conv2d(inputs = X, num_outputs = output_channels,\
                kernel_size = lbc_size, stride = 1, padding = "SAME", activation_fn = None)
        X2 = tf.contrib.layers.batch_norm(X1)
        X3 = tf.nn.relu(X2)

    X_in = X3
    for i in range(conv_layers):
        with tf.name_scope("Lbcnn"):
            X_new = lbcnn(prev_input = X_in, lbc_size = lbc_size, \
                lbc_channels = lbc_filters, output_channels = output_channels, sparsity = sparsity)
            X_in = X_new

    # average pool
    with tf.name_scope("AvgPool"):
        Z = tf.nn.avg_pool(value = X_in, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = "SAME")

    # flatter
    with tf.name_scope("Flatter"):
        P = tf.contrib.layers.flatten(Z)

    # add dropout
    with tf.name_scope("Dropout"):
        F1 = tf.nn.dropout(x = P, keep_prob = Keep_probability)

    # fully connected 1
    with tf.name_scope("Fully"):
        F1 = tf.contrib.layers.fully_connected(inputs = F1, num_outputs = fc_hidden_units, activation_fn = tf.nn.relu)

    # add dropout
    with tf.name_scope("Dropout"):
        F2 = tf.nn.dropout(x = F1, keep_prob = Keep_probability)

    # fully connected 2
    with tf.name_scope("Fully"):
        F2 = tf.contrib.layers.fully_connected(inputs = F2, num_outputs = output_classes, activation_fn = None)

    return F2

def compute_cost(A, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = A))

def CNN_training(X_train, Y_train, X_test, Y_test, options):

    ### parameters for MNIST
    learning_rate = options['LEARNING_RATE'] #1e-3
    epoches = options['EPOCHES'] #80
    batch_size = options['BATCH_SIZE'] #10
    output_classes = options['OUTPUT_CLASSES'] #10
    keep_prob = options['KEEP_PROB'] #0.5
    ### 
    
    m, height, width, input_channels = X_train.shape
    
    # Initializer
    X, Y, Keep_prob = placeholder_initializer(height, width, input_channels, output_classes)
    
    # Model
    A = model(X, Keep_probability = Keep_prob, options = options)

    
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


    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoches):
            mini_batches = batch_shuffle(X_train, Y_train, batch_size)
            num_minibatches = int(m / batch_size) + int(m % batch_size != 0)

            for batch_count, mini_batch in enumerate(mini_batches):
                X_batch, Y_batch = mini_batch

                _, opti_cost = sess.run([optimizer, cost], feed_dict = {X: X_batch, Y: Y_batch, Keep_prob: keep_prob})

                print("|Epoches:  [%d][%d|%d]  Cost: %f" %(i, batch_count+1, num_minibatches, opti_cost))

            # test
            total_test_batches = int(X_test.shape[0]/batch_size)
            accumu_acc= 0
            for test_batch in range(total_test_batches):
                acc = sess.run(accuracy, feed_dict = {X: X_test[test_batch*batch_size:test_batch*batch_size+batch_size,:], Y: Y_test[test_batch*batch_size:test_batch*batch_size+batch_size,:], Keep_prob: 1})
                print("|Test:  [%d][%d|%d] Accuracy: %f\n"  %(i, test_batch+1, total_test_batches, acc))
                accumu_acc += acc


            print("Total Accuracy: %f" %(accumu_acc/total_test_batches))

if __name__ == "__main__":
    # mnist_load
    X_train, Y_train, X_test, Y_test = mnist_load()

    options = {
    #training
    'LEARNING_RATE' : 1e-3,
    'EPOCHES' : 80,
    'BATCH_SIZE' : 10,
    'KEEP_PROB' : 0.5,
    # model
    'LBC_FILTERS' : 512,
    'LBC_SIZE' : 3,
    'IDENTITY_FILTERS' : 16,
    'SPARSITY' : 0.5,
    'CONV_LAYERS' : 75,
    'FC_HIDDEN_UNITS' : 128,
    # dataset
    'OUTPUT_CLASSES' : 10
}
    CNN_training(X_train, Y_train, X_test, Y_test, options)

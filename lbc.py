import numpy as np
import tensorflow as tf
import time
import math

from dataset.mnist_load import mnist_load
from dataset.cifar_load import cifar_load
from dataset.preprocess import cifar_preprocess, mnist_preprocess

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

def batch_generate(X, Y, batch_size):
    [m, n_H, n_W, c] = X.shape

    mini_batches = []

    number_of_batches = int(m / batch_size)

    for i in range(number_of_batches):
        batch_X = X[i*batch_size : i*batch_size + batch_size,:,:,:]
        batch_Y = Y[i*batch_size : i*batch_size + batch_size,:]
        mini_batches.append((batch_X, batch_Y))

    if m % batch_size != 0:
        batch_X_final = X[number_of_batches*batch_size : m,:,:,:]
        batch_Y_final = Y[number_of_batches*batch_size : m,:]
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
    w = np.zeros((number_elements,1))
    w = w.astype(np.float32)
    index = np.random.random_integers(0, number_elements-1, math.floor(number_elements * sparsity))
    
    # we now multiply each 1 in original w with each number 1 or -1 in bnl
    for i in index:
        w[i] = np.random.binomial(1, 0.5, None)*2-1
    
    #print('---')
    #print(np.sum(w == 0))
    #print(np.sum(w == 1))
    #print(np.sum(w == -1))
    
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
    prev_channels = prev_input.shape[3].value
    weight = generate_lbc_weights(height = lbc_size, width = lbc_size, \
        prev_output_channels = prev_channels, number_of_filters = lbc_channels, sparsity = sparsity)

    # batch normalization
    shortcut = tf.identity(prev_input)
    B = tf.contrib.layers.batch_norm(prev_input)
    print(B)
    
    # LBCNN conv
    #Z = tf.nn.conv2d(input = B, filter = weight, strides = [1, 1, 1, 1], padding = "SAME")
    Z = tf.contrib.layers.conv2d(inputs = B, num_outputs = lbc_channels, kernel_size = 3, stride = 1, padding = "SAME", activation_fn = None, weights_initializer = tf.initializers.uniform_unit_scaling())
    A1 = tf.nn.relu(Z)
    print(Z)
    
    # the second step is 1x1 convolution
    A2 = tf.contrib.layers.conv2d(inputs = A1, num_outputs = output_channels, kernel_size = 1, \
            stride = 1, padding = "SAME", activation_fn = None, weights_initializer = tf.initializers.uniform_unit_scaling())
    
    # add input and output, like Resnet
    A3 = tf.add(A2, shortcut)
    print(A3)

    return A3, shortcut, Z, A1, A3

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

    # conv1
    for i in range(conv_layers):
        with tf.name_scope("conv"):
            X_out,BB,ZZ,AA,AAA = lbcnn(prev_input = X_in, lbc_size = lbc_size, \
                lbc_channels = lbc_filters, output_channels = output_channels, sparsity = sparsity)

            X_in = X_out

    # average pool
    with tf.name_scope("AvgPool"):
        Z = tf.nn.avg_pool(value = X_in, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = "VALID")

    # flatter
    with tf.name_scope("Flatter"):
        P = tf.contrib.layers.flatten(Z)

    # add dropout
    with tf.name_scope("Dropout"):
        F1 = tf.nn.dropout(x = P, keep_prob = Keep_probability)

    # fully connected 1
    with tf.name_scope("Fully"):
        F2 = tf.contrib.layers.fully_connected(inputs = F1, num_outputs = fc_hidden_units, activation_fn = tf.nn.relu)

    # add dropout
    with tf.name_scope("Dropout"):
        F3 = tf.nn.dropout(x = F2, keep_prob = Keep_probability)

    # fully connected 2
    with tf.name_scope("Fully"):
        F4 = tf.contrib.layers.fully_connected(inputs = F3, num_outputs = output_classes, activation_fn = None)


    return F4,

def compute_cost(A, Y):
    l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = A) + l2_loss)

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
    A, X1, X2, X3, X4, BB1, ZZ1, AA1, AA2, BB2, ZZ2, AA3, AA4 = model(X, Keep_probability = Keep_prob, options = options)

    # Pre processing data
    if options['DATASET'] == 'mnist':
        X_train = mnist_preprocess(X_train, "train")
        X_test = mnist_preprocess(X_test, "test")
        pass
    elif options['DATASET'] == 'cifar':
        X_train = cifar_preprocess(X_train, "train")
        X_test = cifar_preprocess(X_test, "test")
        pass
    
    cost = compute_cost(A, Y)

    optimizer = tf.train.MomentumOptimizer(learning_rate = 1e-4, momentum = 0.9).minimize(cost)

    # top-1 accuracy
    y_pred = tf.nn.softmax(A)
    y_true = tf.argmax(Y, axis = 1)
    with tf.name_scope("Top1-Accuracy"):
        top1_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = y_pred, targets = y_true, k = 1), tf.float32))

    with tf.name_scope("Top5-Accuracy"):
        top5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions = y_pred, targets = y_true, k = 5), tf.float32))
    
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1

    top_accuracy = 0 # if any acc is better than this, save the model
    saver = tf.train.Saver()
    i = 0
    with tf.Session(config = config) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('./graph/', sess.graph)
        for epoch in range(epoches):

            # train
            mini_batches = batch_shuffle(X_train, Y_train, batch_size)
            num_minibatches = int(m / batch_size) + int(m % batch_size != 0)

            for batch_count, mini_batch in enumerate(mini_batches):
                a_time = time.time()
                X_batch, Y_batch = mini_batch

                _, opti_cost= sess.run([optimizer, cost], feed_dict = {X: X_batch, Y: Y_batch, Keep_prob: keep_prob})
                top_1, top_5 = sess.run([top1_accuracy, top5_accuracy], feed_dict = {X: X_batch, Y: Y_batch, Keep_prob: 1})
                b_time = time.time()


                print(" | Epoches:  [%d][%d|%d]   Time: %5.5f  Cost: %3.5f  top1 %7.3f  top5 %7.3f" %(epoch, batch_count+1, num_minibatches, b_time - a_time, opti_cost, top_1*100, top_5*100))


            # test
            total_test_batches = int(X_test.shape[0]/batch_size) + int(X_test.shape[0] % batch_size !=0)
            mini_batches = batch_generate(X_test, Y_test, batch_size)
            top1Sum = 0
            top5Sum = 0
            for batch_count, mini_batch in enumerate(mini_batches):
                X_batch, Y_batch = mini_batch
                
                top_1, top_5 = sess.run([top1_accuracy, top5_accuracy], feed_dict = {X: X_batch, Y: Y_batch, Keep_prob: 1})
                top1Sum += top_1
                top5Sum += top_5

                print(" | Test:  [%d][%d|%d]  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)" %(epoch, batch_count+1, total_test_batches, top_1*100, top1Sum / (batch_count+1) * 100, top_5*100, top5Sum / (batch_count+1) * 100))
           
            saved_text = "Epoch %d, top1 %7.3f top5 %7.3f \n" %(epoch, top1Sum / total_test_batches * 100, top5Sum / total_test_batches * 100)
            print("Finish epoch %d" %(epoch))
            print(saved_text)

            f = open("result.txt", "a+")
            f.write(saved_text)
            f.close()

            if (top1Sum/total_test_batches) > top_accuracy:
                top_accuracy = top1Sum/total_test_batches
                saver.save(sess, "./result/model_11.ckpt")


        writer.close()

if __name__ == "__main__":
    # mnist_load
    X1_train, Y1_train, X1_test, Y1_test = cifar_load()
    X2_train, Y2_train, X2_test, Y2_test = mnist_load()

    cifar_options = {
    #training
    'LEARNING_RATE' : 1e-4,
    'EPOCHES' : 80,
    'BATCH_SIZE' : 5,
    'KEEP_PROB' : 0.5,
    # model
    'LBC_FILTERS' : 704,
    'LBC_SIZE' : 3,
    'IDENTITY_FILTERS' : 384,
    'SPARSITY' : 0.000001,
    'CONV_LAYERS' : 50,
    'FC_HIDDEN_UNITS' : 512,
    # dataset
    'OUTPUT_CLASSES' : 10,
    "DATASET": "cifar"
}

    mnist_options = {
    #training
    'LEARNING_RATE' : 1e-4,
    'EPOCHES' : 80, #80
    'BATCH_SIZE' : 10,
    'KEEP_PROB' : 0.5,
    # model
    'LBC_FILTERS' : 512, #512
    'LBC_SIZE' : 3,
    'IDENTITY_FILTERS' : 16, #16
    'SPARSITY' : 1,
    'CONV_LAYERS' : 75, #75
    'FC_HIDDEN_UNITS' : 128, #128
    # dataset
    'OUTPUT_CLASSES' : 10,
    "DATASET": "mnist"
}
    CNN_training(X1_train, Y1_train, X1_test, Y1_test, cifar_options)

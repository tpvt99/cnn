import numpy
import tensorflow as tf

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
    I = tf.get_variable("I" + str(layer_no), shape = [1, 1, lbc_channels, output_channels], dtype=tf.float32)
    A2 = tf.nn.conv2d(input = A1, filter = I, strides = [1, 1, 1, 1], padding = "SAME")
    
    # add input and output, like Resnet
    A2 = tf.add(A2, prev_input)

    return A2

def model(X, options):

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
            X_new = lbcnn(prev_input = X_in, layer_no = i, lbc_size = lbc_size, \
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
        F1 = tf.nn.dropout(x = P, keep_prob = 0.5)

    # fully connected 1
    with tf.name_scope("Fully"):
        F1 = tf.contrib.layers.fully_connected(inputs = F1, num_outputs = fc_hidden_units, activation_fn = tf.nn.relu)

    # add dropout
    with tf.name_scope("Dropout"):
        F2 = tf.nn.dropout(x = F1, keep_prob = 0.5)

    # fully connected 2
    with tf.name_scope("Fully"):
        F2 = tf.contrib.layers.fully_connected(inputs = F2, num_outputs = output_classes, activation_fn = None)

    return F2

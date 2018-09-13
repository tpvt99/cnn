import numpy as np
import tensorflow as tf

def generate_lbc_weights(layer_no, height, width, prev_output_channels, number_of_filters, sparsity):
    """
    Arguments:
        layer_no: number of current layer to assign name to weight because tensorflow requires unique name
        height: height of the filter
        width: width of the filter
        prev_output_channels: channels of previous output layers
        number_of_filters: total number of filters (channels)
        sparsity: from 0 to 1
    """
    
    # first step is based on sparsity levels, generate zero and non-zero(1)
    np.random.seed(layer_no)
    number_elements = height * width * prev_output_channels * number_of_filters
    w = np.random.rand(number_elements,1) <= sparsity
    w = w.astype(np.float32)
    
    # we now multiply each 1 in original w with each number 1 or -1 in bnl
    for i in range(w.shape[0]):
        if w[i] == 1:
            w[i] = w[i] * (np.random.binomial(1, 0.5, None)*2-1)
    
    # create the anchor weight, set it to trainable = False to prevent tf train
    w = np.reshape(w, (height, width, prev_output_channels, number_of_filters))

    #out = tf.get_variable("W" + str(layer_no), dtype = tf.float32, initializer = w, trainable = False)

    out = tf.constant(value = w, dtype = tf.float32)

    return out

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def select_highest_activations(Relu):

    print(Relu.shape)

    relu_max = tf.where(tf.equal(Relu, tf.reduce_max(Relu)), Relu, tf.zeros_like(Relu))

    return relu_max

def non_linear_deconv(Relu, W, prev_layer_shape):

    relu_back = tf.nn.relu(Relu)

    print("W shape", W.shape)
    print("prev layer shape", prev_layer_shape)
    print("relu back shape", relu_back.shape)
    deconv = tf.nn.conv2d_transpose(relu_back, W, prev_layer_shape, strides=[1, 1, 1, 1])

    return deconv

def unravel_argmax(argmax, shape, output_list=[]):

    if len(shape) == 2:
        output_list.append(argmax // shape[-1])
        output_list.append(argmax % shape[-1])
        result = tf.stack(output_list)
        return result

    output_list.append(argmax // shape[-1])
    return unravel_argmax(argmax % shape[-1], shape[:-1], output_list)


def unpool(prev_layer, pooling_layer, switches):

    print(switches[0, 0, 0, 0])

    switches = unravel_argmax(switches, prev_layer.get_shape().as_list())

    print(switches)
    print(switches.shape)
    print(switches[:,0,0,0,0])

    unpool = tf.Variable(initial_value=tf.zeros_like(prev_layer))
    pooling_layer_shape = pooling_layer.shape
    for instance in range(pooling_layer_shape[0]):
        for height in range(pooling_layer_shape[1]):
            for width in range(pooling_layer_shape[2]):
                for channel in range(pooling_layer_shape[3]):
                    # print(instance * height * width * channel)
                    index = switches[:, instance, height, width, channel]
                    max_value = pooling_layer[instance, height, width, channel]
                    tf.scatter_nd_update(unpool, tf.reverse(index, [-1]), updates=tf.convert_to_tensor(max_value))

    return unpool

def build_back_model():

    W1 = tf.get_variable(shape=[3, 3, 1, 32], name='W1')
    W2 = tf.get_variable(shape=[3, 3, 32, 64], name='W2')
    W3 = tf.get_variable(shape=[3, 3, 64, 128], name='W3')
    W4 = tf.get_variable(shape=[128 * 4 * 4, 625], name='W4')
    W5 = tf.get_variable(shape=[625, 10], name='W5')


    with tf.name_scope('layer1') as scope:
        # L1 Conv shape=(?, 28, 28, 32)
        #    Pool     ->(?, 14, 14, 32)
        L1, switch1 = build_cnn_layer(X, W1, p_keep_conv)
    with tf.name_scope('layer2') as scope:
        # L2 Conv shape=(?, 14, 14, 64)
        #    Pool     ->(?, 7, 7, 64)
        L2, switch2 = build_cnn_layer(L1, W2, p_keep_conv)
    with tf.name_scope('layer3') as scope:
        # L3 Conv shape=(?, 7, 7, 128)
        #    Pool     ->(?, 4, 4, 128)
        #    Reshape  ->(?, 625)
        reshape = [-1, W4.get_shape().as_list()[0]]
        L3, switch3 = build_cnn_layer(L2, W3, p_keep_conv, reshape=reshape)
    with tf.name_scope('layer4') as scope:
        # L4 FC 4x4x128 inputs -> 625 outputs
        L4 = tf.nn.relu(tf.matmul(L3, W4))
        L4 = tf.nn.dropout(L4, p_keep_hidden)

    # Output(labels) FC 625 inputs -> 10 outputs
    with tf.name_scope('layer5') as scope:
        L5 = tf.matmul(L4, W5, name='model')
        output = tf.nn.softmax(L5, name='output')



def visualize(imgs):

    h = imgs.shape[1]
    w = imgs.shape[2]
    c = imgs.shape[3]

    for img in imgs:

        if c < 3:

            temp_img = np.zeros([h, w, 3])

            for channel in range(c):
                temp_img[:,:,channel] = img[:,:,channel]

            img = temp_img

        plt.imshow(np.abs(img))
        plt.show()

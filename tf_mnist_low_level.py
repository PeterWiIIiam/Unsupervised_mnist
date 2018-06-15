from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import datetime
import os

root_logdir = "tf_logs"
log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))


def model(data, labels, sess, mode='train'):

    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='labels')

    X_shape = X.get_shape().as_list()
    Y_shape = Y.get_shape().as_list()

    conv1 = tf.layers.conv2d(X, 4, kernel_size=(2,2), strides = (1,1), padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, (2,2), strides=(1,1))
    fc1 = tf.layers.dense(tf.contrib.layers.flatten(pool1), units=50, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, units=10, activation=tf.nn.relu)
    output = tf.nn.softmax(fc2)

    loss = tf.losses.softmax_cross_entropy(Y, output)
    loss_summary = tf.summary.scalar('loss', loss)
    file_write = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init) # init needs to at the end

    if mode == 'train':

        for iter in range(2000):
            sess.run(optimizer, feed_dict={X:data,
                                           Y:labels})
            summary_str = loss_summary.eval(feed_dict={X:data,
                                                       Y:labels})
            file_write.add_summary(summary_str, iter)
            
            print(iter)

            saver.save(sess, os.path.join(os.getcwd(), "model_session.ckpt"))

        saver.save(sess, "trained_model")

    if mode == 'test':

        output = sess.run(output, feed_dict={X:data,
                                            Y:labels})

def restore_model(sess):
    saver = tf.train.import_meta_graph('/Users/xhe/Desktop/Unsupervised_mnist.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/Users/xhe/Desktop/'))
    graph = sess.graph
    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    dense_1_kernel = graph.get_tensor_by_name("dense_1/kernel:0")



def load_data():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    return train_data, train_labels, eval_data, eval_labels

def main(argv):

    # train_data, train_labels, eval_data, eval_labels = load_data()
    # X = tf.reshape(train_data, [train_data.shape[0], 28, 28, 1])
    # Y = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=10)
    # print(X.shape, Y.shape)
    #
    # with tf.Session() as sess:
    #     X = X.eval()
    #     Y = Y.eval()
    #     model(X, Y, sess, mode='train')

    with tf.Session() as sess:
        restore_model(sess)
if __name__ == "__main__":
    tf.app.run()
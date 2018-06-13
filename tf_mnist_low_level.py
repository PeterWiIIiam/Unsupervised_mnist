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


def model(data, labels, mode='train'):

    X = tf.placeholder(dtype=tf.float32, shape=[-1, 28, 28, 1])
    Y = tf.placeholder(dtype=tf.float32, shape=[-1, ])
    conv1 = tf.nn.conv2d(X, [2, 2, X.shape[-1], 3], strides=1, padding='SAME')
    pool1 = tf.nn.max_pool(conv1, [conv1.shape[0], 2, 2, conv1.shape[3]], strides=1, padding='VALID')

    fc1 = tf.layers.dense(tf.contrib.layers.flatten(pool1), units=50, activation=tf.nn.relu)
    output = tf.nn.softmax(fc1)

    loss = tf.losses.softmax_cross_entropy(Y, output)
    loss_summary = tf.summary.scalar('loss', loss)
    file_write = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())

    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.Session() as sess:
            for iter in range(2000):
                sess.run(optimizer, feed_dict={X:data,
                                               Y:labels})
                summary_str = loss_summary.eval(feed_dict={X:data,
                                                           Y:data})
                file_write.add_summary(summary_str, iter)


def load_data():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    return train_data, train_labels, eval_data, eval_labels

def main(argv):

    train_data, train_labels, eval_data, eval_labels = load_data()
    X = tf.reshape(train_data, [train_data.shape[0], tf.sqrt(train_data.shape[1]), tf.sqrt(train_data.shape[1]), 1])
    Y = tf.one_hot(indices=tf.cast(train_labels, tf.float16), depth=10)
    model(X, Y, mode='train')

if __name__ == "__main__":
    tf.app.run()
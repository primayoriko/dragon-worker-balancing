'''
A Convolutional Network implementation of MNIST database of handwritten digits 
in Distributed System using TensorFlow library.

Author: Naufal Prima Yoriko

Adapted from
- https://github.com/NTHU-LSALAB/DRAGON 
- https://github.com/aymericdamien/TensorFlow-Examples/ by Aymeric Damien

MNIST source: (http://yann.lecun.com/exdb/mnist/)
'''

from __future__ import print_function
import argparse
import ast
import os
import sys
import time

import tensorflow as tf

# patch older version of tensorflow to use new download mirror
if tf.__version__.startswith('0.12') or tf.__version__.startswith('1.0'):
    import tensorflow.contrib.learn.python.learn.datasets.mnist
    tensorflow.contrib.learn.datasets.mnist.SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

start_time = time.time()

class PrintEvalHook(tf.train.StopAtStepHook):
    def __init__(self, accuracy_step, *args, **kwargs):
        super(PrintEvalHook, self).__init__(*args, **kwargs)
        self.accuracy_step = accuracy_step
        self.start_time = time.time()

    # def after_create_session(self, session, coord):
    #     super(PrintEvalHook, self).after_create_session(session, coord)
    
    def end(self, session):
        sys.stdout.flush()
        print("end hook")
        print("Time elapsed (global): ", time.time() - start_time)
        print("Time elapsed: ", time.time() - self.start_time)
        print("Stop session: ", session._closed)
        if not session._closed:
            print("Test Accuracy:",
                    session.run(self.accuracy_step,
                    feed_dict={ x: mnist.test.images,
                                y: mnist.test.labels,
                                keep_prob: 1.}))
        else:
            print("session somehow stopped")
        sys.stdout.flush()
        super(PrintEvalHook, self).end(session)
        
class Empty:
    pass

FLAGS = Empty()

# Parameters
learning_rate = 0.001
batch_size = 16
display_step = 100
dropout = 0.75  # Dropout, probability to keep units

# Data Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            # Store layers weight & bias
            weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                # 1024 inputs, 10 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([1024, n_classes]))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

            # init var
            init = tf.global_variables_initializer()
            global_step = tf.train.get_or_create_global_step()

            # Construct model
            pred = conv_net(x, weights, biases, keep_prob)

            # Define loss and optimizer
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # loss (?) 
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step) # train_op

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            # hooks = [PrintEvalHook(accuracy), 
            #         tf.train.StopAtStepHook(last_step=FLAGS.global_steps)]
            hooks = [PrintEvalHook(accuracy, last_step=FLAGS.global_steps)]

            with tf.train.MonitoredTrainingSession(master=server.target,
                                        is_chief=(FLAGS.task_index == 0),
                                        config=tf.ConfigProto(
                                            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                                        ),
                                        hooks=hooks) as sess:
                # step = 0
                start_time = time.time()
                sess.run(init)

                while not sess.should_stop():
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, step = sess.run([optimizer, global_step], feed_dict={x: batch_xs, y: batch_ys,
                                                keep_prob: dropout})
                    if step % display_step == 0:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs,
                                                                        y: batch_ys,
                                                                        keep_prob: 1.})
                        print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                            "{:.6f}".format(loss) + ", Train Accuracy: " +
                            "{:.5f}".format(acc))
                        sys.stdout.flush()
                        # if acc >= 0.95:
                        #     break
                    # step += 1
                # print("after session")
                # print("Time elapsed: ", time.time() - start_time)

if __name__ == "__main__":
    TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
    FLAGS.job_name = TF_CONFIG["task"]["type"]
    FLAGS.task_index = TF_CONFIG["task"]["index"]
    FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
    FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
    FLAGS.global_steps = int(os.environ["global_steps"]) if "global_steps" in os.environ else 10000
    tf.app.run(main=main, argv=[sys.argv[0]])

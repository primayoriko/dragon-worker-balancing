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

class Empty:
    pass

FLAGS = Empty()

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 16
display_step = 100
# batch_size = mnist.train.num_examples // 3
# initial_learning_rate = 0.5 
training_epochs = 10
n_hidden = 10
logs_path = "/tmp/mnist/2"

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
# x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

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

def conv_net_test(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # fc3 = tf.nn.relu(fc3)
    # fc3 = tf.nn.dropout(fc3, dropout)
    # out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out

def generate_weights_and_biases_test():
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
        'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        'wd1': tf.Variable(tf.random_normal([8 * 8 * 128, 1024])),
        # 'wd2': tf.Variable(tf.random_normal([1024, 256])),
        # 'wd3': tf.Variable(tf.random_normal([256, 64])),
        # 'out': tf.Variable(tf.random_normal([64, n_classes]))
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([16])),
        'bc2': tf.Variable(tf.random_normal([32])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        # 'bd2': tf.Variable(tf.random_normal([256])),
        # 'bd3': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    return ( weights, biases )

def generate_weights_and_biases():
    # Store layers weight & bias
    # with tf.name_scope("weights"):
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

    # with tf.name_scope("biases"):
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    return ( weights, biases )

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

            # init_op = tf.global_variables_initializer()
            # summary_op = tf.summary.merge_all()
            # global_step = tf.train.get_or_create_global_step()
            global_step = tf.get_variable(
                'global_step',
                [],
                initializer = tf.constant_initializer(0),
                trainable = False)

            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, [None, n_input])
                y = tf.placeholder(tf.float32, [None, n_classes])
                keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

            weights, biases = generate_weights_and_biases()
            pred = conv_net(x, weights, biases, keep_prob)
            # weights, biases = generate_weights_and_biases_test()
            # pred = conv_net_test(x, weights, biases, keep_prob)

            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # cost, loss
            
            with tf.name_scope('train'):
                # grad_op = tf.train.GradientDescentOptimizer(learning_rate)
                grad_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = grad_op.minimize(cross_entropy, global_step=global_step) # train_op

            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            tf.summary.scalar("cost", cross_entropy)
            tf.summary.scalar("accuracy", accuracy)

            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            is_chief = (FLAGS.task_index == 0)

            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            sv = tf.train.Supervisor(
                is_chief=is_chief, global_step=global_step, init_op=init_op)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                start_time = time.time()

                for epoch in range(training_epochs):

                    batch_count = int(mnist.train.num_examples / batch_size)
                    count = 0

                    for i in range(batch_count):
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        
                        _, step = sess.run([train_op,  global_step], 
                                            feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                        writer.add_summary(summary, step)
                        
                        # if step % display_step == 0:
                        #     # Calculate batch loss and accuracy
                        #     loss, acc, summary = sess.run([cross_entropy, accuracy, summary_op], feed_dict={x: batch_x,
                        #                                                     y: batch_y,
                        #                                                     keep_prob: 1.})
                        #     writer.add_summary(summary, step)
                        #     print("Step num: ", step)
                        #     sys.stdout.flush()
                        #     print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                        #         "{:.6f}".format(loss) + ", Training Accuracy= " +
                        #         "{:.5f}".format(acc))
                        #     sys.stdout.flush()

                        count += 1
                        if count % display_step == 0 or i + 1 == batch_count:
                            elapsed_time = time.time() - start_time
                            loss, acc, summary = sess.run([cross_entropy, accuracy, summary_op], feed_dict={x: batch_x,
                                                                            y: batch_y,
                                                                            keep_prob: 1.})
                            print("Step: %d," % (step + 1), 
                                        " Epoch: %2d," % (epoch + 1), 
                                        " Batch: %3d of %3d," % (i + 1, batch_count), 
                                        " Train Acc: %.4f," % acc, 
                                        " Train Loss: %.4f," % loss, 
                                        " AvgTime: %3.2fms" % float(elapsed_time * 1000 / display_step))
                            sys.stdout.flush()
                            writer.add_summary(summary, step)
                            count = 0
                            start_time = time.time()
                
                print("Total Train Time: %3.2fs" % float(time.time() - start_time))
                print("Test Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
                print("Final Loss: %.4f" % loss)

if __name__ == "__main__":
    TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
    FLAGS.job_name = TF_CONFIG["task"]["type"]
    FLAGS.task_index = TF_CONFIG["task"]["index"]
    FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
    FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
    FLAGS.global_steps = int(os.environ["global_steps"]) if "global_steps" in os.environ else 10000
    tf.app.run(main=main, argv=[sys.argv[0]])

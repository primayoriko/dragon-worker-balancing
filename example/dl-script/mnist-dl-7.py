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

# import argparse
import ast
import datetime
import os
import sys
import time

import http.client

import tensorflow as tf

# patch older version of tensorflow to use new download mirror
if tf.__version__.startswith('0.12') or tf.__version__.startswith('1.0'):
    import tensorflow.contrib.learn.python.learn.datasets.mnist
    tensorflow.contrib.learn.datasets.mnist.SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, validation_size=5000)

class Empty:
    pass

FLAGS = Empty()

inf_time_in_seconds = 1000000 

# Parameters
# learning_rate = 1e-3
learning_rate = 5e-4
# learning_rate = 1e-4
# training_iters = 500000
# batch_size = 16
batch_size = 128
display_step = 50

# Network Parameters
n_input = 28 * 28  # MNIST data input (img shape: 28*28) => 784
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.9  # Dropout, probability to keep units

def send_message(message, start_time=None):
    payload = {
            "sender": FLAGS.job_id,
            "worker_index": FLAGS.task_index,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message
        }

    if start_time is not None:
        payload["delta_time"] = time.time() - start_time
    payload = str(payload)
    
    if FLAGS.webhook_link is not "UNKNOWN":
        conn = http.client.HTTPSConnection(FLAGS.webhook_link)
        conn.request("POST", "/", payload, 
            {'Content-Type': 'application/json'})
        time.sleep(2)
        conn.close()

def send_initial_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "job started at {}".format(timestamp)
    send_message(message)

def main(_):
    begin_time = time.time()

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    try:
        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                job_name=FLAGS.job_name,
                                task_index=FLAGS.task_index)
    except tf.errors.InvalidArgumentError as e:
        print("predicted")
        sys.stdout.flush()
        print(str(e))
        sys.stdout.flush()
        time.sleep(inf_time_in_seconds)
    except Exception as e:
        print("predicted, but false type: ", type(e))
        sys.stdout.flush()
        print(str(e))
        sys.stdout.flush()
        send_message(str(e))
        time.sleep(inf_time_in_seconds)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        is_chief = (FLAGS.task_index == 0)

        if is_chief:
            send_initial_timestamp()

        # with tf.device(tf.train.replica_device_setter(
        #     worker_device="/job:worker/task:%d" % FLAGS.task_index,
        #     cluster=cluster)):
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):

            def conv2d(x, W, b, strides=1):
                # Conv2D wrapper, with bias and relu activation
                x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                return tf.nn.relu(x)

            def maxpool2d(x, k=2):
                return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                        padding='SAME')

            def conv_net(x, weights, biases, dropout):
                x = tf.reshape(x, shape=[-1, 28, 28, 1])

                conv1 = conv2d(x, weights['wc1'], biases['bc1'])
                conv1 = maxpool2d(conv1, k=2)

                conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
                conv2 = maxpool2d(conv2, k=2)

                # Reshape conv2 output to fit fully connected layer input
                fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
                fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
                fc1 = tf.nn.relu(fc1)

                fc1 = tf.nn.dropout(fc1, dropout)

                out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

                return out

            # def conv_net_test(x, weights, biases, dropout):
            #     x = tf.reshape(x, shape=[-1, 28, 28, 1])
            #     conv1 = conv2d(x, weights['wc1'], biases['bc1'])
            #     conv1 = maxpool2d(conv1, k=2)
            #     conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
            #     conv2 = maxpool2d(conv2, k=2)

            #     conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
            #     conv3 = maxpool2d(conv3, k=2)
            #     conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
            #     conv4 = maxpool2d(conv4, k=2)

            #     # Reshape conv2 output to fit fully connected layer input
            #     fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
            #     fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            #     # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
            #     # fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
            #     fc1 = tf.nn.relu(fc1)
            #     fc1 = tf.nn.dropout(fc1, dropout)
            #     out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
            #     # fc3 = tf.nn.relu(fc3)
            #     # fc3 = tf.nn.dropout(fc3, dropout)
            #     # out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
            #     return out

            # def generate_weights_and_biases_test():
            #     weights = {
            #         'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
            #         'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
            #         'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            #         'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128])),
            #         'wd1': tf.Variable(tf.random_normal([8 * 8 * 128, 1024])),
            #         # 'wd2': tf.Variable(tf.random_normal([1024, 256])),
            #         # 'wd3': tf.Variable(tf.random_normal([256, 64])),
            #         # 'out': tf.Variable(tf.random_normal([64, n_classes]))
            #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
            #     }

            #     biases = {
            #         'bc1': tf.Variable(tf.random_normal([16])),
            #         'bc2': tf.Variable(tf.random_normal([32])),
            #         'bc3': tf.Variable(tf.random_normal([64])),
            #         'bc4': tf.Variable(tf.random_normal([128])),
            #         'bd1': tf.Variable(tf.random_normal([1024])),
            #         # 'bd2': tf.Variable(tf.random_normal([256])),
            #         # 'bd3': tf.Variable(tf.random_normal([64])),
            #         'out': tf.Variable(tf.random_normal([n_classes]))
            #     }

            #     return ( weights, biases )

            # init var
            init = tf.global_variables_initializer()
            global_step = tf.train.get_or_create_global_step()

            x = tf.placeholder(tf.float32, [None, n_input])
            y = tf.placeholder(tf.float32, [None, n_classes])
            keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

            # Construct model
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
            pred = conv_net(x, weights, biases, keep_prob)
            # weights, biases = generate_weights_and_biases_test()
            # pred = conv_net_test(x, weights, biases, keep_prob)

            # Define loss and optimizer
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # loss (?) 
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step) # train_op

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            total_step = FLAGS.global_steps + 25 * len(worker_hosts)
            hooks = [tf.train.StopAtStepHook(last_step=total_step)]
            # worker_config = tf.ConfigProto()
            worker_config = tf.ConfigProto(
                device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
            )
            with tf.train.MonitoredTrainingSession(master=server.target,
                                        is_chief=is_chief,
                                        config=worker_config,
                                        hooks=hooks) as sess:

                print("tf_config:", os.environ["TF_CONFIG"])
                sys.stdout.flush()

                print("webhook_link:", FLAGS.webhook_link)
                sys.stdout.flush()

                print("job_id:", FLAGS.job_id)
                sys.stdout.flush()

                print("job_name:", FLAGS.job_name)
                sys.stdout.flush()

                print("worker_hosts_init:", worker_hosts)
                sys.stdout.flush()

                sess.run(init)

                flg1, flg2 = (True, True)
                start_time = time.time()

                def test_model():
                    acc = sess.run(accuracy,
                                feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels,
                                            keep_prob: 1.})
                    message = "Testing Accuracy: {:.3f} %".format(acc * 100)
                    print(message)
                    sys.stdout.flush()
                    if is_chief:
                        send_message(message, start_time)
                
                def output_final_time():
                    end_time = time.time()
                    message = "Net time elapsed: {:.3f} s | Gross time elapsed: {:.3f} s".format(
                        end_time - start_time, end_time - begin_time)
                    print(message)
                    sys.stdout.flush()
                    if is_chief:
                        send_message(message)
                
                i = -1
                test_display_step = 10 * display_step
                while not sess.should_stop():
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, step = sess.run([optimizer, global_step], feed_dict={x: batch_xs, y: batch_ys,
                                                keep_prob: dropout})
                    if (step % display_step == 0 or i % display_step == 0) and not sess.should_stop():
                        batch_xs_val, batch_ys_val = mnist.validation.next_batch(batch_size)
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs_val,
                                                                        y: batch_ys_val,
                                                                        keep_prob: 1.})
                        # if not sess.should_stop():
                        #     bc1 = sess.run([biases['bc1']])
                        #     print("Weight 1: ", bc1)
                        #     sys.stdout.flush()
                        # if not sess.should_stop():
                        #     bc2 = sess.run([biases['bc2']])
                        #     print("Weight 2: ", bc2)
                        #     sys.stdout.flush()
                        print("Step num: ", step)
                        sys.stdout.flush()
                        print("i-step num: ", i)
                        sys.stdout.flush()
                        print("Iter " + str(step * batch_size) + ", Minibatch Loss : " +
                            "{:.3f}".format(loss) + ", Validation Accuracy : " +
                            "{:.3f}".format(acc) + " | time: {:.3f} s".format(time.time() - start_time))
                        sys.stdout.flush()
                    
                    if (step % test_display_step == 0 or i % test_display_step == 0) and not sess.should_stop():
                        # test_model()
                        pass

                    if step > FLAGS.global_steps:
                        if flg1:
                            flg1 = False
                            output_final_time()

                        if flg2 and not sess.should_stop():
                            flg2 = False
                            test_model()

if __name__ == "__main__":
    TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
    
    FLAGS.webhook_link = str(os.environ["webhook_link"]) if "webhook_link" in os.environ else "UNKNOWN"
    FLAGS.job_id = str(os.environ["tfjob_id"]) if "tfjob_id" in os.environ else "UNKNOWN"
    FLAGS.job_name = TF_CONFIG["task"]["type"]
    FLAGS.task_index = TF_CONFIG["task"]["index"]
    FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
    FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
    FLAGS.global_steps = int(os.environ["global_steps"]) if "global_steps" in os.environ else 3500
    
    # try:
    #     tf.app.run(main=main, argv=[sys.argv[0]])
    # except Exception as e:
    #     print(e)
    #     sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]])

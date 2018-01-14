# deep-learning
# -*- coding:utf-8 -*-
import input_data
import tensorflow as tf
import input_data
from sys import path
import time
import tkMessageBox
import numpy as np
from PIL import Image,ImageFilter

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


K = 4
L = 8
M = 12
N = 200
X = tf.placeholder("float", [None, 784])

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
B1 = tf.Variable(tf.ones([K]) / 10)

W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L]) / 10)

W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M]) / 10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N]) / 10)

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]) / 10)

pkeep = tf.placeholder(tf.float32)
Y1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(X, [-1, 28, 28, 1]), W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
Y1 = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
Y_ = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y4 = tf.nn.dropout(Y_, pkeep)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(Y))

train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):

    if i == 0:

        fw1 = open("/home/hu/tensorflow/Picture/dataw1.txt", 'w')
        fw1.close()

        fw2 = open("/home/hu/tensorflow/Picture/dataw2.txt", 'w')
        fw2.close()

        fw3 = open("/home/hu/tensorflow/Picture/dataw3.txt", 'w')
        fw3.close()

        fw4 = open("/home/hu/tensorflow/Picture/dataw4.txt", 'w')
        fw4.close()

        fw5 = open("/home/hu/tensorflow/Picture/dataw5.txt", 'w')
        fw5.close()

        fb1 = open("/home/hu/tensorflow/Picture/datab1.txt", 'w')
        fb1.close()

        fb2 = open("/home/hu/tensorflow/Picture/datab2.txt", 'w')
        fb2.close()

        fb3 = open("/home/hu/tensorflow/Picture/datab3.txt", 'w')
        fb3.close()

        fb4 = open("/home/hu/tensorflow/Picture/datab4.txt", 'w')
        fb4.close()

        fb5 = open("/home/hu/tensorflow/Picture/datab5.txt", 'w')
        fb5.close()

    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys, pkeep: 0.75})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        a, c, w1, w2, w3, w4, w5, b1, b2, b3, b4, b5 = sess.run(
            [accuracy, cross_entropy, W1, W2, W3, W4, W5, B1, B2, B3, B4, B5],
            feed_dict={X: mnist.test.images, y_: mnist.test.labels, pkeep: 1.0})
        print i, a
        # save_path = saver.save(sess,"./model/"+str(i+step)+"_train_.ckpt")
        #  save_path = saver.save(sess, "./model/" + str(i + step) + "_train_.ckpt")
        w1 = np.reshape(w1, [-1, 1])
        fw1 = open("/home/hu/tensorflow/Picture/dataw1.txt", 'w')
        for j in w1:
            print >> fw1, "%0.2f" % (float(j))
        fw1.close()
        w2 = np.reshape(w2, [-1, 1])

        fw2 = open("/home/hu/tensorflow/Picture/dataw2.txt", 'w')
        for k in w2:
            print >> fw2, "%0.2f" % (float(k))
        fw2.close()

        w3 = np.reshape(w3, [-1, 1])
        print w3.shape
        fw3 = open("/home/hu/tensorflow/Picture/dataw3.txt", 'w')
        for l in w3:
            print >> fw3, "%0.2f" % (float(l))
        fw3.close()

        w4 = np.reshape(w4, [-1, 1])
        fw4 = open("/home/hu/tensorflow/Picture/dataw4.txt", 'w')
        for z in w4:
            print >> fw4, "%0.2f" % (float(z))
        fw4.close()

        w5 = np.reshape(w5, [-1, 1])
        fw5 = open("/home/hu/tensorflow/Picture/dataw5.txt", 'w')
        for x in w5:
            print >> fw5, "%0.2f" % (float(x))
        fw5.close()

        b1 = np.reshape(b1, [-1, 1])
        fb1 = open("/home/hu/tensorflow/Picture/datab1.txt", 'w')
        for c in b1:
            print >> fb1, "%0.2f" % (float(c))
        fb1.close()

        b2 = np.reshape(b2, [-1, 1])
        fb2 = open("/home/hu/tensorflow/Picture/datab2.txt", 'w')
        for v in b2:
            print >> fb2, "%0.2f" % (float(v))
        fb2.close()

        b3 = np.reshape(b3, [-1, 1])
        fb3 = open("/home/hu/tensorflow/Picture/datab3.txt", 'w')
        for b in b3:
            print >> fb3, "%0.2f" % (float(b))
        fb3.close()

        b4 = np.reshape(b4, [-1, 1])
        fb4 = open("/home/hu/tensorflow/Picture/datab4.txt", 'w')
        for n in b4:
            print >> fb4, "%0.2f" % (float(n))
        fb4.close()

        b5 = np.reshape(b5, [-1, 1])
        fb5 = open("/home/hu/tensorflow/Picture/datab5.txt", 'w')
        for m in b5:
            print >> fb5, "%0.2f" % (float(m))
        fb5.close()

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PrepData import *

BUS_WIDTH = 16;
display_step = 100
learning_rate = 0.01
training_epochs = 1000
beta = 0.005
batch_size = 2048

XY = np.loadtxt('sdataset.txt',dtype = 'int')
np.random.shuffle(XY)
f = XY[:, 0]
l = XY[:, 1]
#
mods = get_mod3_mod5_mod7(f)
f = vec_bin_array(f,BUS_WIDTH)
f = np.c_[f,mods]
f, l = append_bias_reshape(f,l)
n_dim = f.shape[1]

rnd_indices = np.random.rand(len(f)) < 0.70
train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]
fet = f = XY[:, 0]
fet = fet[~rnd_indices]

cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
# W = tf.Variable(tf.ones([n_dim,1]))

weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[n_dim,1], initializer=weight_initer)

init = tf.global_variables_initializer()

regularizer = tf.nn.l2_loss(W)
y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y) + beta * regularizer)
# cost = tf.reduce_mean(tf.square(y_ - Y) )

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for epoch in range(0, len(train_y) // batch_size):
            offset = (epoch * batch_size) % (train_x.shape[0] - batch_size)
            batch_data = train_x[offset:(offset + batch_size), :]
            batch_labels = train_y[offset:(offset + batch_size), :]
            sess.run(training_step, feed_dict={X: batch_data, Y: batch_labels})
            cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: batch_data, Y: batch_labels}))


        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: batch_data, Y: batch_labels})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W))

    # calculate mean square error
    pred_y = sess.run(y_, feed_dict={X: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse = sess.run(mse)
    print("MSE: %.4f" % mse,"RMSE: %.4f" % np.sqrt(mse))
    # plot cost
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y, s = 5)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(fet, pred_y, s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('G(X)')
    plt.show()
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PrepData import *

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

BUS_WIDTH = 16;
display_step = 10
training_epochs = 100
batch_size = 128

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

# Model architecture parameters
n_stocks = n_dim
n_neurons_1 = 4
n_neurons_2 = 4
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))


# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_2, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))


# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))

init = tf.global_variables_initializer()

cost = tf.reduce_mean(tf.square(out - Y))

opt = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(training_epochs):
        shuffle_indices = np.random.permutation(np.arange(len(test_y)))
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]
        for epoch in range(0, len(train_y) // batch_size):
            offset = (epoch * batch_size) % (train_x.shape[0] - batch_size)
            batch_data = train_x[offset:(offset + batch_size), :]
            batch_labels = train_y[offset:(offset + batch_size), :]
            sess.run(opt, feed_dict={X: train_x, Y: train_y})
            cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: batch_data, Y: batch_labels}))

        if (i + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: batch_data, Y: batch_labels})
            print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c))

    # calculate mean square error
    pred_y = sess.run(out, feed_dict={X: test_x})
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
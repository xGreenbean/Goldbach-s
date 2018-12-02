import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PrepData import *
# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

BUS_WIDTH = 16;
display_step = 100
learning_rate = 0.01
training_epochs = 10
beta = 0.005
batch_size = 256

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
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

init = tf.global_variables_initializer()

cost = tf.reduce_mean(tf.square(out - Y))

opt = tf.train.AdamOptimizer().minimize(cost)


# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(test_y)
line2, = ax1.plot(test_y*0.5)
plt.show()

# Number of epochs and batch size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(train_y)))
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]

        # Minibatch training
        for i in range(0, len(train_y) // batch_size):
            start = i * batch_size
            batch_x = train_x[start:start + batch_size]
            batch_y = train_y[start:start + batch_size]
            # Run optimizer with batch
            sess.run(opt, feed_dict={X: batch_x, Y: batch_y})

        cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: batch_x, Y: batch_y}))
            # Show progress
        if np.mod(epoch, 2) == 0:
            c = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))


    # Print final MSE after Training
    mse_final = sess.run(cost, feed_dict={X: test_x, Y: test_y})
    print(mse_final)

    # calculate mean square error
    pred_y = sess.run(out, feed_dict={X: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse = sess.run(cost)
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

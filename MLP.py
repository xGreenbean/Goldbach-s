import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PrepData import *

BUS_WIDTH = 24
#data handeling
XY = np.loadtxt('Bigdataset.txt',dtype = 'int')
np.random.shuffle(XY)
f = XY[:, 0]
l = XY[:, 1]
#

mods = get_mod3_mod5_mod7(f)
f = vec_bin_array(f, BUS_WIDTH)
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

# Parameters
learning_rate = 0.01
training_epochs = 400
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 8 # 1st layer number of features
n_hidden_2 = 4 # 2nd layer number of features
n_input = n_dim # MNIST data input (img shape: 28*28)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 1])
cost_history = np.empty(shape=[1],dtype=float)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        for batch in range(0, len(train_y) // batch_size):
            offset = (epoch * batch_size) % (train_x.shape[0] - batch_size)
            batch_data = train_x[offset:(offset + batch_size), :]
            batch_labels = train_y[offset:(offset + batch_size), :]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_data,
                                                          y: batch_labels})
            cost_history = np.append(cost_history, c)
            # Compute average loss
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: test_x, y: test_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))

    #mse over time graph
    pred_y = sess.run(pred, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse = sess.run(mse)
    print("MSE: %.4f" % mse,"RMSE: %.4f" % np.sqrt(mse))
    # plot cost
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()
    #prdicted , real graph
    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y, s = 5)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    #just a graph
    fig, ax = plt.subplots()
    ax.scatter(fet, pred_y, s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('G(X)')
    plt.show()

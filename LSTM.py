import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PrepData import *

n_hidden_1 = 32  # 1st layer number of features
n_hidden_2 = 8  # 2nd layer number of features
n_hidden_3 = 32  # 3nd layer number of features

BUS_WIDTH = 16  # number of bits required for our binary representation.
batch_size = 100
num_past_numbers = 10   # The number of previous numbers that predict the next partition
step = 15   # how far do we predict
D = 1   # Dimension of our output
beta = 0    # for regularization

# data prepossessing.
XY = np.loadtxt('Datasets/sdataset.txt', dtype='int')
features = XY[:, 0]
labels = XY[:, 1]
mods = get_mod3_mod5_mod7(features)
features = vec_bin_array(features, BUS_WIDTH)
features = np.c_[features, mods]
features, labels = append_bias_reshape(features, labels)
n_dim = features.shape[1]

splits = int(len(XY)*0.8)    # train\test split
train_x = features[:splits]
train_y = labels[:splits]
test_x = features[splits:]
test_y = labels[splits:]
fet = XY[:, 0]
fet = fet[splits:]


num_input_examples_train = int((len(train_x) - (num_past_numbers + 1)) / step)
num_input_examples_test = int((len(test_x) - (num_past_numbers + 1)) / step)


# Get test data


def next_test():
    curr_range_start = 0
    data_x = np.zeros(shape=(num_input_examples_test, num_past_numbers, n_dim))
    data_y = np.zeros(shape=(num_input_examples_test, D))
    for ep in range(num_input_examples_test):
        for inc in range(num_past_numbers):
            data_x[ep][inc] = test_x[curr_range_start + ep * step + inc]
        data_y[ep] = test_y[curr_range_start + ep * step + num_past_numbers]
    return (data_x, data_y)


curr_batch = 0
# Get batch data


def next_batch():
    global curr_batch
    curr_range_start = curr_batch*batch_size
    if curr_range_start + batch_size >= num_input_examples_train:
        return (None, None)
    data_x = np.zeros(shape=(batch_size, num_past_numbers, n_dim))
    data_y = np.zeros(shape=(batch_size,D))
    for ep in range(batch_size):
        for inc in range(num_past_numbers):
            data_x[ep][inc] = train_x[curr_range_start + ep * step + inc]
        data_y[ep] = train_y[curr_range_start + ep * step + num_past_numbers]
    curr_batch += 1
    return (data_x, data_y)

#LSTM
cellsize = 30   #size of lstm cell
x = tf.placeholder(tf.float32, [None, num_past_numbers, n_dim])
y = tf.placeholder(tf.float32, [None, D])

lstm_cell = tf.nn.rnn_cell.LSTMCell(cellsize, forget_bias=0.0)
output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

output = tf.transpose(output, [1, 0, 2])
last = output[-1]

#MLP
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([cellsize, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Construct model
pred = multilayer_perceptron(last, weights, biases)

cost = tf.reduce_mean(tf.square(pred - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

num_of_epochs = 50
for epoch in range(num_of_epochs):
    curr_batch = 0
    loss = 0
    while True:
        batch_xs, batch_ys = next_batch()
        if batch_xs is None:
            break
        else:
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            loss += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
    print("step %d, training loss %g" % (epoch, loss / curr_batch))

curr_batch = 0
# use produce our test data
test_xs, test_ys = next_test()
pred_y = sess.run(pred, feed_dict={x: test_xs})
mse = tf.reduce_mean(tf.square(pred_y - test_ys))
mse = sess.run(mse)
print("MSE: %.4f" % mse,"RMSE: %.4f" % np.sqrt(mse))

plt.plot(pred_y, color='red', label='Prediction')
plt.plot(test_ys, color='blue', label='Ground Truth')
plt.legend(loc='upper left')
plt.show()

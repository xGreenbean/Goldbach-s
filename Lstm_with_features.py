import tensorflow as tf
import numpy as np
import urllib
import matplotlib.pyplot as plt
from PrepData import *
BUS_WIDTH = 16
batch_size = 15
num_past_characters = 10 # The number of previous characters that predict the next character
step = 1
D = 1

BUS_WIDTH = 16
#data handeling
XY = np.loadtxt('Datasets/sdataset.txt',dtype = 'int')
np.random.shuffle(XY)
f = XY[:, 0]
l = XY[:, 1]
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
fet = XY[:, 0]
fet = fet[~rnd_indices]


num_input_examples_train = int((len(train_x)-(num_past_characters+1))/step)
num_input_examples_test = int((len(test_x)-(num_past_characters+1))/step)

curr_batch = 0

def next_test():
    curr_range_start = 0
    data_x = np.zeros(shape=(num_input_examples_test,num_past_characters,n_dim))
    data_y = np.zeros(shape=(num_input_examples_test,D))
    for ep in range(num_input_examples_test):
        for inc in range(num_past_characters):
            data_x[ep][inc] = test_x[curr_range_start + ep * step + inc]
        data_y[ep] = test_y[curr_range_start + ep * step + num_past_characters]
    return (data_x, data_y)

def next_batch():
    global curr_batch
    curr_range_start = curr_batch*batch_size
    if curr_range_start + batch_size >= num_input_examples_train:
        return (None, None)
    data_x = np.zeros(shape=(batch_size,num_past_characters,n_dim))
    data_y = np.zeros(shape=(batch_size,D))
    for ep in range(batch_size):
        for inc in range(num_past_characters):
            data_x[ep][inc] = train_x[curr_range_start + ep * step + inc]
        data_y[ep] = train_y[curr_range_start+ep*step+num_past_characters]
    curr_batch += 1
    return (data_x, data_y)

cellsize = 30
x = tf.placeholder(tf.float32, [None, num_past_characters, n_dim])
y = tf.placeholder(tf.float32, [None, D])

lstm_cell = tf.nn.rnn_cell.LSTMCell(cellsize, forget_bias=0.0)
output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

output = tf.transpose(output, [1, 0, 2])
last = output[-1]

W = tf.Variable(tf.truncated_normal([cellsize, D], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[D]))
z = tf.matmul(last, W) + b

loss = tf.reduce_mean(tf.square(y - z))
optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



num_of_epochs = 5
for ephoch in range(num_of_epochs):
    curr_batch = 0
    cost = 0
    while True:
        batch_xs, batch_ys = next_batch()
        if batch_xs is None:
            break
        else:
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            cost += sess.run(loss,feed_dict={x: batch_xs, y: batch_ys})
    print("step %d, training loss %g"%(ephoch, cost/curr_batch))
test_xs, test_ys = next_test()
pred_y = sess.run(z, feed_dict={x: test_xs})
mse = tf.reduce_mean(tf.square(pred_y[:9700] - test_y[:9700]))
mse = sess.run(mse)
print("MSE: %.4f" % mse,"RMSE: %.4f" % np.sqrt(mse))



def DL():
    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    BUS_WIDTH = 16;

    XY = np.loadtxt('sdataset.txt',dtype = 'int')
    np.random.shuffle(XY)
    f = XY[:, 0]
    l = XY[:, 1]

    mods = get_mod3_mod5_mod7(f)
    f = vec_bin_array(f,BUS_WIDTH)
    f = np.c_[f,mods]
    f, l = append_bias_reshape(f,l)
    n_dim = f.shape[1]

    rnd_indices = np.random.rand(len(f)) < 0.7
    train_x = f[rnd_indices]
    train_y = l[rnd_indices]
    test_x = f[~rnd_indices]
    test_y = l[~rnd_indices]

    n_features = n_dim
    n_neurons_1 = 128
    n_neurons_2 = 64
    n_neurons_3 = 32
    n_neurons_4 = 16
    n_target = 1

    # Placeholder
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,1])

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
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

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Make Session
    net = tf.Session()

    # Run initializer
    net.run(tf.global_variables_initializer())

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(test_y)
    line2, = ax1.plot(test_y*0.5)
    plt.show()

    # Number of epochs and batch size
    epochs = 100
    batch_size = 128

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(test_y)))
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]

        # Minibatch training
        for epoch in range(0, len(train_y) // batch_size):
            offset = (epoch * batch_size) % (train_x.shape[0] - batch_size)
            batch_data = train_x[offset:(offset + batch_size), :]
            batch_labels = train_y[offset:(offset + batch_size), :]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_data, Y: batch_labels})

            # Show progress
            if np.mod(epoch, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: train_x})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(epoch))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(epoch) + '.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)


    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: test_x, Y: test_y})
    print(mse_final)

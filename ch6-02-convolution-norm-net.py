import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import cifar10_input
cifar10_input.maybe_download_and_extract()


learning_rate = 0.01
training_epochs = 20
batch_size = 10
dislay_step = 1


def inputs(eval_data=True):
    data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data=eval_data,
                                data_dir=data_dir,
                                batch_size=batch_size)


def distorted_inputs():
    data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=batch_size)


def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name="moments")
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(
        phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var)
    )

    normed = tf.nn.batch_norm_with_global_normalization(
        x,
        mean,
        var,
        beta,
        gamma,
        1e-3,
        True
    )
    return normed


def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name="moments")
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(
        phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var)
    )

    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(
        x_r,
        mean,
        var,
        beta,
        gamma,
        1e-3,
        True
    )
    return tf.reshape(normed, [-1, n_out])


def conv2d(input, weight_shape, bias_shape, phase_train, visualize=False):
    weight_prod = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_prod)**0.5)
    W = tf.get_variable("W",
                        weight_shape,
                        initializer=weight_init)

    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b",
                        bias_shape,
                        initializer=bias_init)
    logits = tf.nn.bias_add(
        tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'),
        b
    )
    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))


def max_pool(input, k=2):
    return tf.nn.max_pool(input,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='SAME')


def layer(input, weight_shape, bias_shape, phase_train):
    weight_stdev = (2.0/weight_shape[0])**0.5
    weight_init = tf.random_normal_initializer(stddev=weight_stdev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable(
        "W",
        weight_shape,
        initializer=weight_init
    )
    b = tf.get_variable(
        "b",
        bias_shape,
        initializer=bias_init
    )
    logits = tf.matmul(input, W) + b
    return tf.nn.relu(
        layer_batch_norm(logits, weight_shape[1], phase_train)
    )


def inference(x, keep_prob, phase_train):

    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 64], [64], phase_train)
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], phase_train)
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc_1"):
        dim = 1
        for d in pool_2.get_shape()[1:].as_list():
            dim *= d

        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        fc_1 = layer(pool_2_flat, [dim, 384], [384], phase_train)
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("fc_2"):
        fc_2 = layer(fc_1_drop, [384, 192], [192], phase_train)
        # apply dropout
        fc_2_drop = tf.nn.dropout(fc_2, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_2_drop, [192, 10], [10], phase_train)
        return output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation", accuracy)
    return accuracy


with tf.Graph().as_default():
    # cifar10_input data image of shape 24 * 24 * 3 (color)
    x = tf.placeholder(tf.float32, name="x", shape=[None, 24, 24, 3])
    y = tf.placeholder(tf.float32, name="y", shape=[None])
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)
    with tf.variable_scope("mlp_model"):
        output = inference(x, 0.5, phase_train)
    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = training(cost, global_step)
    eval_op = evaluate(output, y)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    distorted_images, distorted_labels = distorted_inputs()
    val_images, val_labels = inputs()

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(
        "ch6-02-logistic_logs/",
        graph_def=sess.graph_def
    )

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    valid_errors = []

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(
            cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size
        )

        # Loop over all batches
        for i in range(total_batch):
            train_x, train_y = sess.run([distorted_images, distorted_labels])
            # fit the training set
            feed_dict = {
                x: train_x,
                y: train_y,
                keep_prob: 1,
                phase_train: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            # computer average loss
            minibach_cost = sess.run(cost, feed_dict=feed_dict)
            avg_cost += minibach_cost / total_batch
        valid_errors.append(avg_cost)

        # displays logs per epoch step
        if epoch % dislay_step == 0:
            val_feed_dict = {
                x: cifar10_input.validation.images,
                y: cifar10_input.validation.labels
            }
            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
            print("Epoch ", epoch, "Validation Error: ", (1 - accuracy))
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))

            save_path = saver.save(
                sess,
                "ch6-02-logistic_logs/model-checkpoint"
            )
            print("Model saved in file: %s" % save_path)

    print("Optimization Finished!")

    test_feed_dict = {
        x: cifar10_input.test.images,
        y: cifar10_input.test.labels
    }
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
    print("Test Accuracy ", accuracy)

    print("plot the results")
    plt.plot(np.arange(0, training_epochs, 1), valid_errors, 'ro')
    plt.ylabel('Error Incurred')
    plt.xlabel('Alpha')
    plt.show()

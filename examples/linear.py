"""Example linear model."""

import tensorflow as tf

if __name__ == "__main__":
    sess = tf.Session()
    W = tf.Variable([[.3, .3, .3], [.3, .3, .3], [.3, .3, .3]], dtype=tf.float32)
    b = tf.Variable([-.3, -.3, -.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)

    model = (W * x) + b

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(model, {x: [1, 2, 3]}))

    y = tf.placeholder(tf.float32)
    error = tf.square(model - y)
    loss = tf.reduce_sum(error)

    print(sess.run(loss, {x: [1, 2, 3], y: [0, 0, 1]}))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    print("Training")
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3], y: [0, 0, 1]})

    print(sess.run(model, {x: [1, 2, 3], y: [0, 0, 1]}))

#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pr_load_dataset as pr

# get signs depth data
tr_sub=['S01']
te_sub=[]
val_sub=[]
signs = pr.read_data_sets('../papermcpr/repo/dataset/*.jpeg',
                   tr_sub, te_sub, val_sub,
                   one_hot=True,
                   dtype=np.uint8,
                   reshape=True)


# Parameters
learning_rate = 0.01
training_epochs = 400
batch_size = 200
display_step = 1
examples_to_show = 5

# Network Parameters
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features
n_input = 65536 # signs data input (img shape: 256*256)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
  'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
  'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
  'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
  'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
  # Encoder Hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                 biases['encoder_b1']))
  # Decoder Hidden layer with sigmoid activation #2
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                 biases['encoder_b2']))
  return layer_2


# Building the decoder
def decoder(x):
  # Encoder Hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                 biases['decoder_b1']))
  # Decoder Hidden layer with sigmoid activation #2
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                 biases['decoder_b2']))
  return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  total_batch = int(signs.train.num_examples/batch_size)
  print(signs.train.num_examples)
  print(batch_size)
  # Training cycle
  for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
      batch_xs, batch_ys = signs.train.next_batch(batch_size)
      # Run optimization op (backprop) and cost op (to get loss value)
      _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

    # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1),
            "cost=", "{:.9f}".format(c))

  print("Optimization Finished!")

  # Applying encode and decode over test set
  encode_decode = sess.run(
    y_pred, feed_dict={X: signs.test.images[:examples_to_show]})
  # Compare original images with their reconstructions
  f, a = plt.subplots(2, 10, figsize=(10, 2))
  for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(signs.test.images[i], (256, 256)))
    a[1][i].imshow(np.reshape(encode_decode[i], (256, 256)))
  f.show()
  plt.draw()
  plt.waitforbuttonpress()



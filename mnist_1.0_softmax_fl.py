# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
#fully connected layer (softmax)       W [784, 10]     b[10]
#                                      Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([28 * 28, 10]))
b = tf.Variable(tf.zeros([10]))
W_fl = tf.Variable(tf.zeros([28 * 28, 10]))
b_fl = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
Y_fl = tf.nn.softmax(tf.matmul(XX, W_fl) + b_fl)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 100.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10
tf.summary.scalar('loss_ce', cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy_ce', accuracy)
train_ce = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

#fl = Y_ * tf.log(Y_fl) *tf.pow(1 - Y_fl, 2) #2
ce = Y_ * tf.log(Y_fl)
fl = ce * tf.pow(1 - Y_fl, 2)
focal_loss = -tf.reduce_mean(fl) * 1000.0
tf.summary.scalar('loss_fl', focal_loss)
correct_prediction_fl = tf.equal(tf.argmax(Y_fl, 1), tf.argmax(Y_, 1))
accuracy_fl = tf.reduce_mean(tf.cast(correct_prediction_fl, tf.float32))
tf.summary.scalar('accuracy_fl', accuracy_fl)

train_fl = tf.train.GradientDescentOptimizer(0.005).minimize(focal_loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter('./log', sess.graph) #write to file
merge_op = tf.summary.merge_all()

# You can call this function in a loop to train the model, 100 images at a time
plt.ion()
x_axis = []
y_ce_axis = []
y_fl_axis = []
for step in range(100):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    sess.run([train_ce, train_fl], feed_dict={X: batch_X, Y_: batch_Y})
    #train_a_ce, train_a_fl, train_loss_ce, train_loss_fl, result= sess.run([accuracy,accuracy_fl, cross_entropy, focal_loss, merge_op], feed_dict={X: batch_X, Y_: batch_Y})
    train_loss_ce, train_loss_fl= sess.run([cross_entropy, focal_loss], feed_dict={X: batch_X, Y_: batch_Y})
    test_a_ce, test_a_fl, test_loss_ce, test_loss_fl, result = sess.run([accuracy,accuracy_fl, cross_entropy, focal_loss, merge_op], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    #print str(step), "train : accuracy_ce:" + str(train_a_ce) + "accuracy_fl:" + str(train_a_fl) + " loss_ce: " + str(train_loss_ce)  + " loss_fl: " + str(train_loss_fl)
    print str(step), "train : loss_ce:" + str(train_loss_ce) + "loss_fl:"+ str(train_loss_fl)
    print "test : accuracy_ce:" + str(test_a_ce) + "accuracy_fl:" + str(test_a_fl) + " loss_ce: " + str(test_loss_ce) + " loss_fl: " + str(test_loss_fl)
    writer.add_summary(result, step)

    x_axis.append(step)
    y_ce_axis.append(test_a_ce)
    y_fl_axis.append(test_a_fl)

plt.plot(x_axis, y_ce_axis, 'go',linewidth=2,label = "ce")
plt.plot(x_axis, y_fl_axis, 'ro',linewidth=2,label = "fl")
plt.legend()
plt.ioff()
plt.show()

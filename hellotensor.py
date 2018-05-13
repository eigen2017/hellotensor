import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

# read data from csv
train_data = pd.read_csv("iris_training.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])
test_data = pd.read_csv("iris_test.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])

# encode results to onehot
train_data['f5'] = train_data['f5'].map({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]})
test_data['f5'] = test_data['f5'].map({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]})

# separate train data
train_x = train_data[['f1', 'f2', 'f3', 'f4']]
train_y = train_data.ix[:, 'f5']

# separate test data
test_x = test_data[['f1', 'f2', 'f3', 'f4']]
test_y = test_data.ix[:, 'f5']

# placeholders for inputs and outputs
X = tf.placeholder(tf.float32, [None, 4], name='my_X')
Y = tf.placeholder(tf.float32, [None, 3], name='my_Y')

# weight and bias
weight = tf.Variable(tf.zeros([4, 3]), name='my_weight')
bias = tf.Variable(tf.zeros([3]), name='my_bias')

# output after going activation function
Z = tf.add(tf.matmul(X, weight), bias, name='my_Z')
output = tf.nn.softmax(Z, name='my_output')
# cost funciton
squres = tf.square(Y - output, name='my_squres')
cost_pre = tf.reduce_mean(squres, axis=1, name='my_cost_pre')
cost = tf.reduce_mean(cost_pre, axis=0, name='my_cost')
# train model
optimizer = tf.train.AdamOptimizer(0.01, name='my_optimizer')
train = optimizer.minimize(cost, name='my_train')
# tf.train.Optimizer
# check sucess and failures
# success = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
# # calculate accuracy
# accuracy = tf.reduce_mean(tf.cast(success, tf.float32)) * 100

# initialize variables
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# start the tensorflow session
with tf.Session(config=config) as sess:
    tf.summary.FileWriter("logs/", sess.graph)
    costs = []
    sess.run(init)
    # train model 1000 times
    for i in range(1000):
        _, c = sess.run([train, cost], {X: train_x, Y: [t for t in train_y.as_matrix()]})
        costs.append(c)

    print("Training finished!")

    # plot cost graph
    plt.plot(range(1000), costs)
    plt.title("Cost Variation")
    plt.show()
    # print("Accuracy: %.2f" % accuracy.eval({X: test_x, Y: [t for t in test_y.as_matrix()]}))

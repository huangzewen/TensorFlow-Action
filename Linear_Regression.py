import tensorflow as tf
import numpy as np

money=np.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
click=np.array([[11], [8], [8], [6],[ 7], [7], [7], [8], [9], [5]]).astype(np.float32)
X_test = money[0:5].reshape(-1, 1)
Y_test = click[0:5]
X_train = money[5:].reshape(-1, 1)
Y_train = click[5:]
x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_sum(tf.pow((y_-y), 2))
train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_history = []
for i in range(100):
    feed = {x: X_train, y_: Y_train}
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))
    print("After %d interation:" %i)
    print("W: %f" %sess.run(W))
    print("b: %f" %sess.run(b))
    print("cost: %f" %sess.run(cost, feed_dict=feed))
    
print("W_val: %f, b_val: %f, cost_val: %f"  %(sess.run(W), sess.run(b), sess.run(cost, feed_dict=feed)))

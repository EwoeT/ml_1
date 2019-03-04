

import tensorflow as tf
import numpy as np
import csv
import numpy as np
import scipy as sp
from sklearn import model_selection, datasets, metrics

y_t = []
train_target = []
X_t=np.genfromtxt("/home/ewoe/Downloads/tensorflow-material/data/train-data.csv",delimiter=",")
data_test=np.genfromtxt("/home/ewoe/Downloads/tensorflow-material/data/test-data.csv",delimiter=",")

with open ('/home/ewoe/Downloads/tensorflow-material/data/train-target.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		train_target.append(row)

for tr in train_target:
	y_t.append(tr[0])

#encoding data
from sklearn import preprocessing
values = np.array(y_t)

le = preprocessing.LabelEncoder()
integer_encoded = le.fit_transform(values)
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_t = onehot_encoder.fit_transform(integer_encoded)


#splitting data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)



#modelling algorithm
X = tf.placeholder(tf.float32, [None, 128])
y = tf.placeholder(tf.float32, [None, 26])

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(X, W):
  return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

X_image = tf.reshape(X, [-1, 16, 8, 1])

h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv)
out = tf.transpose(y_hat)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#training over full training data
sess.run(tf.global_variables_initializer())
for i in range(860):
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={X: X_t[40*i:40*(i+1)], y: y_t[40*i:40*(i+1)], keep_prob: 1.0})
		print('step {}, training accuracy {}'.format(i, train_accuracy))
	train_step.run(feed_dict={X: X_t[40*i:40*(i+1)], y: y_t[40*i:40*(i+1)], keep_prob: 0.5})


pred = sess.run(out, feed_dict={X: data_test, keep_prob:1.0})



indexf = tf.argmax(pred)
nwf=[]
ch = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
newwwf= sess.run(indexf)
for row in newwwf:
    nwf.append(row)
for ypf in nwf:
    print(ch[ypf])
newwwf.shape


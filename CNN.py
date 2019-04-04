import pandas as pd
import numpy as np
import pickle
import time
import tensorflow as tf
import matplotlib.pyplot as plt


def load_file(filename_):  # 定义数据导入函数
    with open(filename_, 'rb') as fo:
        data_ = pickle.load(fo, encoding='latin1')
    return data_


def conv2d(x, w):  # 定义卷积层操作
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):  # 定义池化层操作
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):  # 定义卷积核权重变量
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):  # 定义卷积核偏置变量
    return tf.Variable(tf.constant(0.1, shape=shape))


train_data = {'data': [], 'labels': []}  # 训练和测试数据集设置为字典格式，其中data为10000*3072的numpy向量，labels为10000*1的列表
test_data = {'data': [], 'labels': []}

for i in range(1, 6):  # 训练集导入数据
    filename = "data_batch_" + str(i)
    data = load_file(filename)
    train_data['data'] += list(data['data'])
    train_data['labels'] += data['labels']

filename = "test_batch"  # 测试集导入数据
data = load_file(filename)
test_data['data'] += list(data['data'])
test_data['labels'] += data['labels']

Label_size = 10  # 定义一些常量
Hidden_layer_size = 512
Max_pixel_value = 255
Batch_size = 50
Max_iteration = 2000
Print_interval = 100
Test_NUM = 200
Lambda = 0.001

x = tf.placeholder(tf.float32, [None, 3072])
y_ = tf.placeholder(tf.float32, [None, Label_size])
x_image = tf.reshape(x, [-1, 32, 32, 3])

w_conv1 = weight_variable([5, 5, 3, 32])  # 第一层卷积和池化
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])  # 第二层卷积和池化
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

w_fc1 = weight_variable([8*8*64, Hidden_layer_size])  # 全连接层1
b_fc1 = bias_variable([Hidden_layer_size])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)  # 对全连接层dropout,防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([Hidden_layer_size, Label_size])  # 全连接层2
b_fc2 = bias_variable([Label_size])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)  # 输出层用softmax实现多分类

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 计算预测准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # 计算交叉熵损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

x_train = np.array(train_data['data']) / Max_pixel_value  # 将输入数据做归一化处理，将输出数据进行one-hot编码
y_train = np.array(pd.get_dummies(train_data['labels']))
x_test = np.array(test_data['data']) / Max_pixel_value
y_test = np.array(pd.get_dummies(test_data['labels']))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
model_path = "E:\deep learning"
save_path = saver.save(sess, model_path)

total_accuracy = np.zeros(int(Max_iteration/Print_interval)+1)
total_loss = np.zeros(int(Max_iteration/Print_interval)+1)

time1 = time.clock()

for i in range(Max_iteration+1):
    index = i * Batch_size % 50000  # 每次迭代都用一个Batch的数据进行训练
    train_step.run(feed_dict={x: x_train[index: index + Batch_size],
                              y_: y_train[index: index + Batch_size], keep_prob: 0.8})  # [0,0.2,0.5,0.8]
    if i % Print_interval == 0:  # 每迭代100次在测试集上计算准确度和在训练集上计算交叉熵损失
        accuracy_value = accuracy.eval(feed_dict={x: x_test[0: Test_NUM],
                                                  y_: y_test[0: Test_NUM], keep_prob: 1.0})
        total_accuracy[int(i/Print_interval)] = accuracy_value
        loss_value = cross_entropy.eval(feed_dict={x: x_train[index: index + Batch_size],
                                                   y_: y_train[index: index + Batch_size], keep_prob: 1.0})
        total_loss[int(i / Print_interval)] = loss_value
        print("step %d,training accuracy %g,loss %g" % (i, accuracy_value, loss_value))
    if i == Max_iteration:  # 在测试集上的最终预测准确度
        test_accuracy = accuracy.eval(feed_dict={x: x_test[0: 1000], y_: y_test[0: 1000], keep_prob: 1.0})
        print("test accuracy %g" % test_accuracy)

time2 = time.clock()
diff_time = time2-time1
print(diff_time)

# 画准确度和误差图
x = []
for i in range(int(Max_iteration/Print_interval)+1):
    x.append(i)
y = total_accuracy
plt.xlabel('Test Number')
plt.ylabel('Accuracy')
plt.plot(x, y, label='Frist line', linewidth=1, color='r', marker='o', markerfacecolor='red', markersize=2)
plt.show()
plt.clf()

x = []
for i in range(int(Max_iteration/Print_interval)+1):
    x.append(i)
y = total_loss
plt.xlabel('Test Number')
plt.ylabel('Loss')
plt.plot(x, y, label='Frist line', linewidth=1, color='b', marker='o', markerfacecolor='blue', markersize=2)
plt.show()
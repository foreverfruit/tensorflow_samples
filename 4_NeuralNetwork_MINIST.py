# -*- coding: utf-8 -*-
# @File    : 4_NeuralNetwork_MINIST.py
# @Time    : 2018/5/21 10:56
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 简单的两层网络实现的手写体识别,所用网络是一个多层感知机，是一种前向结构的人工神经网络

import input_data
import tensorflow as tf

# 1.数据载入，该部分和之前logisticRegression一样
minist = input_data.read_data_sets('data/', one_hot=True)
train_x = minist.train.images
train_y = minist.train.labels
test_x = minist.test.images
test_y = minist.test.labels
print("train shape:", train_x.shape, train_y.shape)
print("test  shape:", test_x.shape, test_y.shape)
print("----------MNIST loaded----------------")

# 2.建立网络模型
'''
此处采用简单的两层网络
input：n*784
layer1：256个neuron，w：784*256，b：256
layer2：128个neuron，w：256*128，b：128
output：10个类别，w：128*10，b：10
activation：sigmod
cost：cross_entropy
'''
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_class = 10

# 输入与输出
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 网络参数
# 权重以高斯分布初始化
w = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), dtype=tf.float32),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), dtype=tf.float32),
    'w_out': tf.Variable(tf.random_normal([n_hidden_2, n_class]), dtype=tf.float32)
}
# 偏置都采用0值初始化
b = {
    'b1': tf.Variable(tf.zeros([n_hidden_1]), dtype=tf.float32),
    'b2': tf.Variable(tf.zeros([n_hidden_2]), dtype=tf.float32),
    'b_out': tf.Variable(tf.zeros([n_class]), dtype=tf.float32)
}
print('NeuralNetwork Ready!')


def forward_propagation(_x, _w, _b):
    """
    网络的前向传播
    :param _x:网络输入
    :param _w:网络的各层权重
    :param _b:网络各层偏置
    :return: 网络的输出
    """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _w['w1']), _b['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _w['w2']), _b['b2']))
    # 输出层不用激活
    return tf.add(tf.matmul(layer_2, _w['w_out']), _b['b_out'])

# 3.优化求解loss，更新参数
y_ = forward_propagation(x, w, b)
# 平均loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
lr = 0.01
optm = tf.train.GradientDescentOptimizer(lr).minimize(cost)
result = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

# 4.训练
epochs = 40
batch_size = 100
batch_count = int(minist.train.num_examples / batch_size)
display_step = 4

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('---------- TRAINING -------------')
for epoch in range(epochs):
    # 每一轮训练
    arg_cost = 0.
    # 这里提前声明，抑制下文if block中的提示
    batch_x, batch_y = None, None
    for batch_index in range(batch_count):
        batch_x, batch_y = minist.train.next_batch(batch_size)
        feed = {x: batch_x, y: batch_y}
        sess.run(optm, feed_dict=feed)
        arg_cost += sess.run(cost, feed_dict=feed)
    arg_cost /= batch_count
    # display
    if epoch % display_step == display_step - 1:
        # 这里模型在训练集上的准确率，可以用所有训练样本，也可以用最后一个训练批次样本，这里用后者
        feed_train = {x: batch_x, y: batch_y}
        feed_test = {x: test_x, y: test_y}
        ac_train = sess.run(accuracy, feed_dict=feed_train)
        ac_test = sess.run(accuracy, feed_dict=feed_test)
        print('Epoch: %03d/%03d cost: %.5f train_accuray:%0.5f test_accuray:%0.5f' % (
            epoch+1, epochs, arg_cost, ac_train, ac_test))

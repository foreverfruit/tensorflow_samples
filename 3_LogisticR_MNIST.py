# -*- coding: utf-8 -*-
# @File    : 3_LogisticR_MNIST.py
# @Time    : 2018/5/20 12:21
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 逻辑回归实现的MNIST数据集手写体识别任务

import input_data
import numpy as np
import tensorflow as tf

'''
MNIST数据集：此处采用tensorflow sample所带的mnist数据集的预处理脚本，input_data.py
实现了数据的读取，向量化。
number of trian data is 55000
number of test data is 10000
每个图片28*28=784维
10分类
'''

# 1.数据读取
'''
one_hot:一种映射编码方式
特征并不总是连续值，而有可能是分类值。比如星期类型，有星期一、星期二、……、星期日
若用[1,7]进行编码，求距离的时候周一和周日距离很远（7），这不合适。
故周一用[1 0 0 0 0 0 0],周日用[0 0 0 0 0 0 1],这就是one-hot编码
对于离散型特征，基于树的方法是不需要使用one-hot编码的，例如随机森林等。
基于距离的模型，都是要使用one-hot编码，例如神经网络等。
'''
minist = input_data.read_data_sets('data/', one_hot=True)
train_x = minist.train.images
train_y = minist.train.labels
test_x = minist.test.images
test_y = minist.test.labels
print("----------MNIST loaded----------------")
print("train shape:", train_x.shape, train_y.shape)
print("test  shape:", test_x.shape, test_y.shape)

# 2.建立逻辑回归模型
# 输入数据，None表示第一维不固定，可理解为占位
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 参数矩阵(注意维度对齐)
# w = tf.Variable(tf.random_normal([784, 10]), name='w')
w = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([10]), name='b')
# 预测，softmax(y_ = x*w+b),通过softmax将输出变成10个类的得分
y_ = tf.nn.softmax(tf.matmul(x, w) + b)
# cost,这里逻辑回归，softmax，采用对数损失函数，是-log(真实类别概率(得分))
# reduction_indices=1表示对哪个轴求和，0是行，1为列，这里按列(结果为列)求和，已知每一个样本得到一行结果
# 求和就是每一个样本的真实分类概率的和，结合logistic regression的对数损失函数定义理解
# 在reduce_mean，则求所有样本的平均cost
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=1))
# optimize
lr = 0.01
optm = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# 3.训练
# argmax:Returns the index with the largest value across dimensions of a tensor
# equal:Returns the truth value of (x == y) element-wise,returns A `Tensor` of type `bool`.
result = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
# cast:将bool的result向量转成float的，然后求mean，得出准确率
accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 训练参数
'''
batchsize：在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练
iteration：迭代，1个iteration等于使用batchsize个样本训练一次
epoch：迭代次数，1个epoch等于使用训练集中的全部样本训练一次
       一个epoch = 所有训练样本的一个正向传递和一个反向传递
'''
training_epochs = 50
batch_size = 100
display_step = 5

# MINI-BATCH
print('------ TRAINING -----------')
for epoch in range(training_epochs):
    avg_cost = 0.
    # 这样分批是丢弃掉不整除的那部分余数数据,num_batch 批次数目
    num_batch = int(minist.train.num_examples / batch_size)
    for i in range(num_batch):
        # 每一趟epoch中的一个batch训练，这里默认batch中数据会shuffle
        batch_x, batch_y = minist.train.next_batch(batch_size)
        feeds = {x: batch_x, y: batch_y}
        # 训练
        sess.run(optm, feed_dict=feeds)
        # 训练结果，这里run返回的是cost值，即当前batch_size样本的平均cost
        avg_cost += sess.run(cost, feed_dict=feeds)
        # print(cost.eval())
    # 平均cost
    avg_cost /= num_batch

    # 打印训练log
    if epoch % display_step == 0:
        # 分别求测试集和训练集上的准确率
        # 这里是求accuracy，只有前向过程，没有反向最小化梯度过程，此时模型经过optm已经得到了w和b
        feeds_train = {x: train_x, y: train_y}
        feeds_test = {x: test_x, y: test_y}
        train_acc = sess.run(accuracy, feed_dict=feeds_train)
        test_acc = sess.run(accuracy, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))

print('---------DONE------------')

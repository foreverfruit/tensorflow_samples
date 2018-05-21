# -*- coding: utf-8 -*-
# @File    : 2_LinearRegression.py
# @Time    : 2018/5/18 16:59
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : tensorflow实现的线性回归——二次曲线

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ------数据准备-------------------
# 生成1000个数据对作为训练集
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = 0.9 * x1 * x1 + 0.6 * x1 + 0.2 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x = np.array([v[0] for v in vectors_set])
y = np.array([v[1] for v in vectors_set])

# ----------训练------------------
# 生成参数
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
C = tf.Variable(tf.zeros([1]))

y_ = A * x * x + B * x + C
# 定义loss
loss = tf.reduce_mean(tf.square(y - y_))
# 梯度下降法优化参数(学习率为lr)
lr = 0.1
optimizer = tf.train.GradientDescentOptimizer(lr)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss)

# 通过session执行上述操作
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# 初始化 A B C
print('A=', sess.run(A), ' B=', sess.run(B), ' C=', sess.run(C), ' loss=', sess.run(loss))

# 执行训练，根据loss阈值退出迭代
step = 0
while (sess.run(loss) > 1e-3):
    step += 1
    sess.run(train)
    # 输出训练好的ABC
    print('step=', step, ' A=', sess.run(A), ' B=', sess.run(B), ' C=', sess.run(C), ' loss=', sess.run(loss))

# 原图
plt.scatter(x, y, c='r')
x = np.linspace(x.min(),x.max(),100)
temp = sess.run(A) * x * x + sess.run(B) * x + sess.run(C)
# 拟合曲线
plt.plot(x, temp, c='g')
plt.show()

# -*- coding: utf-8 -*-
# @File    : 5_CNN_MINIST.py
# @Time    : 2018/5/22 7:46
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 建立简单的卷积神经网络结构，实现手写体的识别
import input_data
import tensorflow as tf

# 1.数据准备，不再赘述，参照之前的sample
minist = input_data.read_data_sets('data/', one_hot=True)
train_x = minist.train.images
train_y = minist.train.labels
test_x = minist.test.images
test_y = minist.test.labels
print("train shape:", train_x.shape, train_y.shape)
print("test  shape:", test_x.shape, test_y.shape)
print("----------MNIST loaded----------------")

# 2.搭建卷积神经网络结构
n_input = 784
n_out = 10
# 标准差
std = 0.1
w = {
    'w_conv1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=std)),
    'w_conv2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=std)),
    # 此处的7*7是前两层卷积池化之后的尺寸，是通过网络结构参数，计算出来的
    'w_fc1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=std)),
    'w_fc2': tf.Variable(tf.random_normal([1024, n_out], stddev=std))
}
b = {
    'b_conv1': tf.Variable(tf.zeros([64])),
    'b_conv2': tf.Variable(tf.zeros([128])),
    'b_fc1': tf.Variable(tf.zeros([1024])),
    'b_fc2': tf.Variable([tf.zeros([n_out])])
}


def forward_propagation(_x, _w, _b, _keepratio):
    """
    卷积神经网络前向传播结构
    :param _x: 输入
    :param _w: 各层权值
    :param _b: 各层偏置
    :param _keepratio: 保留节点率(=1-失活率)
    :return: 输出
    """
    # 1.将输入x向量矩阵化，因为卷积要在图像矩阵上操作
    _x = tf.reshape(_x, [-1, 28, 28, 1])
    # 2.第一个卷积+池化+dropout
    conv1 = tf.nn.conv2d(_x, _w['w_conv1'], strides=[1, 1, 1, 1], padding='SAME')
    # 输出归一化（保证下一层的输入数据是经过归一化的）
    # _mean, _var = tf.nn.moments(conv1, [0, 1, 2])
    # conv1 = tf.nn.batch_normalization(conv1, _mean, _var, 0, 1, 0.0001)
    # 激活函数，activation
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b['b_conv1']))
    # 推荐数据流动过程中查看shape变化情况
    print('conv1:', conv1.shape)
    # 池化
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('pool1:', pool1.shape)
    # 失活，dropout
    out_conv1 = tf.nn.dropout(pool1, _keepratio)

    # 3.第二个卷积+池化+dropout
    conv2 = tf.nn.conv2d(out_conv1, _w['w_conv2'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(conv1, [0, 1, 2])
    # conv1 = tf.nn.batch_normalization(conv1, _mean, _var, 0, 1, 0.0001)
    # 激活函数，activation
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b['b_conv2']))
    print('conv2:', conv2.shape)
    # 池化
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('pool2:', pool2.shape)
    # 失活，dropout
    out_conv2 = tf.nn.dropout(pool2, _keepratio)

    # 4.向量化，之后的全连接的输入应该是一个样本的特征向量
    feature_vector = tf.reshape(out_conv2, [-1, _w['w_fc1'].get_shape().as_list()[0]])
    print('feature_vector:', feature_vector.shape)

    # 5.第一个 full connected layer，特征向量降维
    fc1 = tf.nn.relu(tf.add(tf.matmul(feature_vector, _w['w_fc1']), _b['b_fc1']))
    fc1_do = tf.nn.dropout(fc1, _keepratio)
    print('fc1:', fc1.shape)

    # 6.第二个 full connected layer，分类器
    out = tf.add(tf.matmul(fc1_do, _w['w_fc2']), _b['b_fc2'])
    print('fc2:', out.shape)
    return out


print('------- CNN Ready! Start Training----------')

# 3.优化求解最小化cost
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_out])
keepratio = tf.placeholder(tf.float32)

y_ = forward_propagation(x, w, b, keepratio)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
lr = 0.001
optm = tf.train.AdamOptimizer(lr).minimize(cost)
result_bool = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(result_bool, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 4.训练
epochs = 10
display_step = 1
# 本人是cpu版本tensorflow，全部样本训练太慢，这里选用部分数据，且batch_size也较小
batch_size = 100
# batch_count = 100
batch_count = int(minist.train.num_examples / batch_size)

for epoch in range(epochs):
    avg_cost = 0.
    batch_x, batch_y = None, None
    for batch_index in range(batch_count):
        batch_x, batch_y = minist.train.next_batch(batch_size)
        feeds = {x: batch_x, y: batch_y, keepratio: 0.6}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost /= batch_count

    if epoch % display_step == display_step - 1:
        feed_train = {x: batch_x, y: batch_y, keepratio: 0.6}
        # 改成tensorflow-gpu版本后这里一次test所有测试集会显存溢出。简单起见，这里只用前200个测试数据
        # feed_test = {x: test_x, y: test_y, keepratio: 1}
        feed_test = {x: test_x[:200], y: test_y[:200], keepratio: 1}
        ac_train = sess.run(accuracy, feed_dict=feed_train)
        ac_test = sess.run(accuracy, feed_dict=feed_test)
        print('Epoch: %03d/%03d cost: %.5f train_accuray:%0.5f test_accuray:%0.5f' % (
            epoch + 1, epochs, avg_cost, ac_train, ac_test))

print('------- COMPLETE ------------')

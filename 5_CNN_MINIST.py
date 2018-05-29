# -*- coding: utf-8 -*-
# @File    : 5_CNN_MINIST.py
# @Time    : 2018/5/29 20:49
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 建立简单的卷积神经网络结构，实现手写体的识别

import input_data
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 数据集合
def load_dataset():
    print('------------ DATA LOADING ------------')
    # minist数据集采用tensorflow自带的加载脚本input_data.py加载
    minist = input_data.read_data_sets('data/', one_hot=True)
    # 训练集55000，验证集5000，测试集10000
    # 模型训练的时候，验证集全部输入，我显卡会OOM，所以只用了前1000张做准确率的测试
    # 测试的时候，没有采用数据集自带的sample，而是自己预处理了手写的数据用于测试
    print("train shape:", minist.train.images.shape, minist.train.labels.shape)
    print("test  shape:", minist.test.images.shape, minist.test.labels.shape)
    print("valid shape:", minist.validation.images.shape, minist.validation.labels.shape)
    print("----------MNIST loaded----------------")
    return {'train': minist.train, 'test': minist.test, 'valid': minist.validation}


# 定义网络结构，预定义网络参数
def net_params():
    # 网络中的参数（输入，输出，权重，偏置）
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
    print('-------------- CNN_NET READY! --------------')
    return {'input_len': n_input, 'output_len': n_out, 'weight': w, 'bias': b}


# 前向传播过程。（各网络层的连接）
def forward_propagation(_x, _w, _b, _keep_ratio):
    # 1.将输入x向量矩阵化，因为卷积要在图像矩阵上操作
    _x = tf.reshape(_x, [-1, 28, 28, 1])
    # 2.第一个卷积+池化+dropout
    conv1 = tf.nn.conv2d(_x, _w['w_conv1'], strides=[1, 1, 1, 1], padding='SAME')
    # 输出归一化（保证下一层的输入数据是经过归一化的）
    # _mean, _var = tf.nn.moments(conv1, [0, 1, 2])
    # conv1 = tf.nn.batch_normalization(conv1, _mean, _var, 0, 1, 0.0001)
    # 激活函数，activation
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, _b['b_conv1']))
    # 推荐数据流动过程中查看shape变化情况
    print('conv1:', conv1.shape)
    # 池化
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('pool1:', pool1.shape)
    # 失活，dropout
    out_conv1 = tf.nn.dropout(pool1, _keep_ratio)

    # 3.第二个卷积+池化+dropout
    conv2 = tf.nn.conv2d(out_conv1, _w['w_conv2'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(conv1, [0, 1, 2])
    # conv1 = tf.nn.batch_normalization(conv1, _mean, _var, 0, 1, 0.0001)
    # 激活函数，activation
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, _b['b_conv2']))
    print('conv2:', conv2.shape)
    # 池化
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('pool2:', pool2.shape)
    # 失活，dropout
    out_conv2 = tf.nn.dropout(pool2, _keep_ratio)

    # 4.向量化，之后的全连接的输入应该是一个样本的特征向量
    feature_vector = tf.reshape(out_conv2, [-1, _w['w_fc1'].get_shape().as_list()[0]])
    print('feature_vector:', feature_vector.shape)

    # 5.第一个 full connected layer，特征向量降维
    fc1 = tf.nn.relu(tf.add(tf.matmul(feature_vector, _w['w_fc1']), _b['b_fc1']))
    fc1_do = tf.nn.dropout(fc1, _keep_ratio)
    print('fc1:', fc1.shape)

    # 6.第二个 full connected layer，分类器
    out = tf.add(tf.matmul(fc1_do, _w['w_fc2']), _b['b_fc2'])
    print('fc2:', out.shape)
    return out


# 训练过程
def training(train, valid):
    # 先得到网络参数
    params = net_params()
    n_input = params['input_len']
    n_out = params['output_len']
    w = params['weight']
    b = params['bias']
    keep_ratio = tf.placeholder(tf.float32)

    # 输入数据，及其对应的真实labels，这里用placeholder占位
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_out])

    y_ = forward_propagation(x, w, b, keep_ratio)
    # 训练任务：优化求解最小化cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
    lr = 0.01
    optm = tf.train.AdamOptimizer(lr).minimize(cost)
    # 计算准确率，用以衡量模型
    result_bool = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(result_bool, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('------- Start Training! ----------')
    # 训练参数
    epochs = 20
    display_step = 1
    # 本人是cpu版本tensorflow，全部样本训练太慢，这里选用部分数据，且batch_size也较小
    batch_size = 200
    # batch_count = 100
    batch_count = int(train.num_examples / batch_size)

    for epoch in range(epochs):
        avg_cost = 0.
        batch_x, batch_y = None, None
        # 分批次进行最小化loss的训练过程
        for batch_index in range(batch_count):
            batch_x, batch_y = train.next_batch(batch_size)
            feeds = {x: batch_x, y: batch_y, keep_ratio: 0.6}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost /= batch_count

        # 检查当前模型的准确率，查看拟合情况
        if epoch % display_step == display_step - 1:
            feed_train = {x: batch_x, y: batch_y, keep_ratio: 1}
            # 全部的验证集都会OOM，所以只取前1000张
            feed_valid = {x: valid.images[:1000], y: valid.labels[:1000], keep_ratio: 1}
            ac_train = sess.run(accuracy, feed_dict=feed_train)
            ac_valid = sess.run(accuracy, feed_dict=feed_valid)
            print('Epoch: %03d/%03d cost: %.5f train_accuray:%0.5f valid_accuray:%0.5f' % (
                epoch + 1, epochs, avg_cost, ac_train, ac_valid))

    print('------- TRAINING COMPLETE ------------')
    # 保存模型
    saver = tf.train.Saver()
    # 这里后缀名ckpt，表示checkpoint，这个可以任意
    save_path = saver.save(sess, 'model/cnn_mnist_model.ckpt')
    print(save_path)
    print('------ MODEL SAVED --------------')


def image_preprocess(file_path: str):
    """
    读取图片并返回图片的预处理之后的数据，预处理包括（resize、reshape、threshold）
    :param file_path: 图片的地址
    :return: numpy.ndarray类型的数据预处理之后的数据
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 默认双线性插值resize
    image = cv2.resize(image, (28, 28))
    # 二值化
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # 为了强化图像，再进行腐蚀和膨胀操作(这里因为经过反向二值化，应该做闭操作)
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # 显示图片（查看二值效果）
    # plt.imshow(image)
    # plt.show()
    # 归一化
    image = image / 255.
    image = image.astype(np.float32)
    return np.reshape(image, [1, 784])


def test():
    # 先得到网络参数
    params = net_params()
    n_input = params['input_len']
    n_out = params['output_len']
    w = params['weight']
    b = params['bias']
    keep_ratio = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_out])

    y_ = forward_propagation(x, w, b, keep_ratio)
    # 求具体的分类结果
    result = tf.argmax(tf.nn.softmax(y_), 1)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 加载模型
    saver = tf.train.Saver()
    saver.restore(sess, 'model/cnn_mnist_model.ckpt')
    print('---------- Model restored! ------------')
    # 进行测试
    print('---------- TESTING ---------------------')
    # 输入图像
    img_path = 'data/mini_test_2_original.png'
    test_2 = image_preprocess(img_path)
    img_path = 'data/mini_test_3_original.png'
    test_3 = image_preprocess(img_path)
    img_path = 'data/mini_test_7_original.png'
    test_7 = image_preprocess(img_path)
    img_path = 'data/mini_test_8.png'
    test_8 = image_preprocess(img_path)

    # 组成输入矩阵
    test_input = np.vstack((test_2, test_3, test_7, test_8))
    # 输入网络计算
    feed_test = {x: test_input, keep_ratio: 1}
    result = sess.run(result, feed_dict=feed_test)
    print('------------ 测试结果 --------------')
    for i, v in enumerate(result):
        print('第%d个输入的是：%d' % (i + 1, v))

    # 查看具体的得分情况
    print(sess.run(y_, feed_dict=feed_test))


if __name__ == '__main__':
    #data_set = load_dataset()
    #training(data_set['train'], data_set['valid'])
    test()
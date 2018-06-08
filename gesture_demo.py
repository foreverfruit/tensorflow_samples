# -*- coding: utf-8 -*-
# @File    : gesture_demo.py
# @Time    : 2018/6/4 14:36
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : cnn识别手势图片

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

gesture_path = os.getcwd() + os.sep + 'gesture' + os.sep + 'train'
dict_dir_label = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
length, width = 64, 64


# 使用二值化模型
def binary_mask(gray):
    minValue = 70
    # 高斯模糊:高斯滤波器中像素的权重与其距中心像素的距离成比例
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    # 图像的二值化提取目标,动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res


def image_preprocess(file_path: str):
    # 局部自适应二值化
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 默认双线性插值resize
    image = cv2.resize(image, (width, length))
    image = binary_mask(image)

    image = cv2.bitwise_not(image)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(image, cmap=plt.cm.gray)
    plt.show()

    image = image.astype('float32')
    # 归一化
    image = image / 255.
    image = image.flatten()
    return image


def get_img_list_path(train_set_path, im_list):
    """
    得到所有图片的路径
    """
    all_path = os.listdir(train_set_path)
    for f in all_path:
        p = os.path.join(train_set_path, f)
        if os.path.isdir(p):
            get_img_list_path(p, im_list)
        elif os.path.isfile(p):
            if os.path.splitext(p)[1] == '.png':
                im_list.append(p)


def get_data_set(valid_ratio=0.2):
    img_list = []
    img_path = gesture_path
    get_img_list_path(img_path, img_list)

    data_ = pd.DataFrame(columns=('image', 'label'))

    # 根据路径读图
    for i, path_ in enumerate(img_list):
        dir_ = os.path.basename(os.path.split(path_)[0])
        label_ = dict_dir_label[dir_]
        img_ = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
        img_ = cv2.resize(img_, (width, length))
        img_ = cv2.bitwise_not(img_)
        img_ = img_.astype('float32')
        # 归一化
        img_ = img_ / 255.
        img_ = img_.flatten()
        data_.loc[i] = [img_, label_]

    index = data_.index.tolist()
    np.random.seed(123)
    np.random.shuffle(index)

    train_size = int(len(index) * (1 - valid_ratio))
    train_ = data_.loc[index[:train_size]]
    valid_ = data_.loc[index[train_size:]]
    np.random.seed()

    train_x_, train_y_ = [], []
    for i in train_.index:
        label_value = train_.loc[i, 'label']
        label_ = np.zeros(6, np.int32)
        label_[label_value] = 1
        train_y_.append(label_)

        img_ = train_.loc[i, 'image']
        train_x_.append(img_)

    train_x_ = np.vstack(train_x_)
    train_y_ = np.vstack(train_y_)

    valid_x_, valid_y_ = [], []
    for i in valid_.index:
        label_value = valid_.loc[i, 'label']
        label_ = np.zeros(6, np.int32)
        label_[label_value] = 1
        valid_y_.append(label_)

        img_ = valid_.loc[i, 'image']
        valid_x_.append(img_)

    valid_x_ = np.vstack(valid_x_)
    valid_y_ = np.vstack(valid_y_)

    return train_x_, train_y_, valid_x_, valid_y_


# 定义网络结构，预定义网络参数
def net_params():
    # 网络中的参数（输入，输出，权重，偏置）
    n_input = width * length
    n = int(width / 4)
    n_out = 6
    # 标准差
    std = 0.1
    w = {
        'w_conv1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=std)),
        'w_conv2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=std)),
        # 此处的7*7是前两层卷积池化之后的尺寸，是通过网络结构参数，计算出来的
        'w_fc1': tf.Variable(tf.random_normal([n * n * 128, 1024], stddev=std)),
        'w_fc2': tf.Variable(tf.random_normal([1024, n_out], stddev=std))
    }
    b = {
        'b_conv1': tf.Variable(tf.zeros([64])),
        'b_conv2': tf.Variable(tf.zeros([128])),
        'b_fc1': tf.Variable(tf.zeros([1024])),
        'b_fc2': tf.Variable(tf.zeros([n_out]))
    }
    print('-------------- CNN_NET READY! --------------')
    return {'input_len': n_input, 'output_len': n_out, 'weight': w, 'bias': b}


# 前向传播过程。（各网络层的连接）
def forward_propagation(_x, _w, _b, _keep_ratio):
    # 1.将输入x向量矩阵化，因为卷积要在图像矩阵上操作
    _x = tf.reshape(_x, [-1, width, length, 1])
    # 2.第一个卷积+池化+dropout
    conv1 = tf.nn.conv2d(_x, _w['w_conv1'], strides=[1, 1, 1, 1], padding='SAME')
    # 输出归一化（保证下一层的输入数据是经过归一化的）
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
def training(train_x, train_y, valid_x, valid_y):
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
    lr = 0.0001
    optm = tf.train.AdamOptimizer(lr).minimize(cost)
    # 计算准确率，用以衡量模型
    result_bool = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(result_bool, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('------- Start Training! ----------')
    # 训练参数
    epochs = 40
    display_step = 1
    # 本人是cpu版本tensorflow，全部样本训练太慢，这里选用部分数据，且batch_size也较小
    batch_size = 80
    # batch_count = 100
    batch_count = int(len(train_x) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0.
        batch_x, batch_y = None, None
        # 分批次进行最小化loss的训练过程
        for batch_index in range(batch_count):
            batch_x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size, :]
            batch_y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size, :]
            feeds = {x: batch_x, y: batch_y, keep_ratio: 0.6}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost /= batch_count

        # 检查当前模型的准确率，查看拟合情况
        if epoch % display_step == display_step - 1:
            feed_train = {x: batch_x, y: batch_y, keep_ratio: 1}
            feed_valid = {x: valid_x, y: valid_y, keep_ratio: 1}
            ac_train = sess.run(accuracy, feed_dict=feed_train)
            ac_valid = sess.run(accuracy, feed_dict=feed_valid)
            print('Epoch: %03d/%03d cost: %.5f train_accuray:%0.5f valid_accuray:%0.5f' % (
                epoch + 1, epochs, avg_cost, ac_train, ac_valid))

    print('------- TRAINING COMPLETE ------------')
    # 保存模型
    # saver = tf.train.Saver()
    # # 这里后缀名ckpt，表示checkpoint，这个可以任意
    # save_path = saver.save(sess, 'model/cnn_gesture_model.ckpt')
    # print(save_path)
    # print('------ MODEL SAVED --------------')


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
    saver.restore(sess, 'model/cnn_gesture_model.ckpt')
    print('---------- Model restored! ------------')
    # 进行测试
    print('---------- TESTING ---------------------')
    # 输入图像
    img_path = 'gesture/test/1.jpg'
    # img_path = 'gesture/test/zero.png'
    test_1 = image_preprocess(img_path)
    img_path = 'gesture/test/2.jpg'
    # img_path = 'gesture/test/one.png'
    test_2 = image_preprocess(img_path)
    img_path = 'gesture/test/3.jpg'
    # img_path = 'gesture/test/three.png'
    test_3 = image_preprocess(img_path)
    img_path = 'gesture/test/4.jpg'
    # img_path = 'gesture/test/four.png'
    test_4 = image_preprocess(img_path)
    img_path = 'gesture/test/5.jpg'
    # img_path = 'gesture/test/five.png'
    test_5 = image_preprocess(img_path)

    # 组成输入矩阵
    test_input = np.vstack((test_1, test_2, test_3, test_4,test_5))
    print(test_input.shape)
    # 输入网络计算
    feed_test = {x: test_input, keep_ratio: 1}
    result = sess.run(result, feed_dict=feed_test)
    print('------------ 测试结果 --------------')
    for i, v in enumerate(result):
        print('第%d个输入的是：%d' % (i + 1, v))

    # 查看具体的得分情况
    # print(sess.run(y_, feed_dict=feed_test))


if __name__ == '__main__':
    train_x_, train_y_, valid_x_, valid_y_ = get_data_set()
    # 验证图像与label
    # print(valid_x_.shape,valid_y_.shape)
    # x_ = train_x_[1].reshape([width, length])
    # y_ = train_y_[1]
    # print(type(y_))
    # print(x_)
    # cv2.imshow('image', x_)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # training(train_x_, train_y_, valid_x_, valid_y_)
    test()
    pass
